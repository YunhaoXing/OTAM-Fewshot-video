import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import sklearn
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import OTAM
from transforms import *
from opts import parser

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == 'something':
        num_class = 5
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 5
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    model = OTAM(num_class, args.num_segments, args.modality,
            base_model=args.arch, dropout=args.dropout, partial_bn=not args.no_partialbn)
   

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args,"", args.train_list,
                   new_length=data_length,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                   train_augmentation,
                   Stack(roll=args.arch == 'BNInception'),
                   ToTorchFormatTensor(div=args.arch != 'BNInception'),
                   normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args,"", args.val_list,
                   new_length=data_length,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    #if args.loss_type == 'nll':
    #    criterion = torch.nn.CrossEntropyLoss().cuda()
    #else:
    #    raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            validate(val_loader, model, (epoch + 1) * len(train_loader))


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    end = time.time()
    for i, (support, query) in enumerate(train_loader):
        print(i)
        support_output = {}
        data_time.update(time.time() - end)
        # compute output
        loss_tot = 0
        for x_query in query:
            #print(x_query)
            input_var = torch.autograd.Variable(x_query[0])      
            out_query = model(input_var)
            gt_pair = []
            tot = 0
            dis_tot = 0
            for x_support in support:
                input_var = torch.autograd.Variable(x_support[0]).cuda()
                out_support = model(input_var)    
                if(x_support[1]==x_query[1]):             
                    gt_pair.append(get_min_distance(out_query,out_support))
                dis_tot = dis_tot + torch.exp(-get_min_distance(out_query,out_support))   
            loss = 0
            for x_ in gt_pair:
                tot +=1
                loss += -torch.log((torch.exp(-x_))/dis_tot)
            loss_tot += loss.item()
            loss = loss/tot
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # measure elapsed time
        print(loss_tot)

        batch_time.update(time.time() - end)
        end = time.time()

def validate(val_loader, model, iter, logger=None):
    batch_time = AverageMeter()
    gt = []
    pred = []
    model.eval()
    correct = 0
    tot = 0
    for i,(support, query) in enumerate(val_loader):
       for x_query in query:
            #print(x_query)
            input_var = (x_query[0]).cuda()
            with torch.no_grad():
                out_query = model(input_var)
            gt.append(x_query[1])
            
            min_dis = 10000

            for x_support in support:
                with torch.no_grad():
                    input_var = (x_support[0]).cuda()
                    out_support = model(input_var)
                dis = get_min_distance(out_query,out_support)
                if(dis<min_dis):
                    min_dis = dis
                    min_label = x_support[1]
            pred.append(min_label) 
    for x in range(len(gt)):
       if (gt[x]==pred[x]):
            correct +=1
       tot+=1              
    print(tot,(correct/tot))
    end = time.time()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

def calculate_dis(mar_x,mar_y):
    return 1-((mar_x*mar_y).sum())/(((mar_x*mar_x).sum())*((mar_y*mar_y).sum()))
 
def calculate_min_distance(tensor_x, tensor_y, rev=True):

    pad = torch.nn.ZeroPad2d(padding=(1, 1, 0, 0))
    pad2 = torch.nn.ConstantPad2d((0, 0, 1, 0),100)
    T = args.num_segments
    dis_mat = torch.Tensor(T,T)
    for l in range(T):
        for m in range(T):
            if(rev == True):
                dis_mat[l][m] = calculate_dis(tensor_x[l],tensor_y[m])
            else:
                dis_mat[T-l-1][T-m-1] = calculate_dis(tensor_x[l],tensor_y[m])
    #print(dis_mat)
    distance0 = pad(dis_mat)
    distance = pad2(distance0)
    opt = torch.zeros(T,T+2)
    for l in range(T):
        for m in range(T+2):
            if m==1 or m==T+1:
                opt[l][m] = min(opt[l-1][m-1],opt[l-1,m],opt[l][m-1])+distance[l][m]
            else:
                opt[l][m]= min(opt[l-1][m-1], opt[l,m-1]) + distance[l][m]
    return opt[T-1][T+1]
    
def get_min_distance(tensor_x,tensor_y):
    return (calculate_min_distance(tensor_x,tensor_y)+calculate_min_distance(tensor_x,tensor_y,rev=False))/2    
    
def get_loss(gt_x, gt_y, support_set):
    dis_tot = 0
    for s in support_set:
        dis_tot = dis_tot + torch.exp(-get_min_distance(gt_x,s))
    return -torch.log((torch.exp(-get_min_distance(gt_x,gt_y)))/dis_tot)
     
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']



if __name__ == '__main__':
    main()
