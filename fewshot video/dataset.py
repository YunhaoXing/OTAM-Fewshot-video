import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from torchvision import transforms

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        base_path = "/home/yhxing/fewshot video/something/rawframes/"
        return base_path + self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, args, root_path, list_file, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False,
                 fix_seed=True,n_aug_support_samples=1):

        self.root_path = root_path
        self.list_file = list_file
        self.new_length = new_length
        self.image_tmpl = image_tmpl
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.n_aug_support_samples = n_aug_support_samples
        self.fix_seed = fix_seed
        self.transform = transform
        
        self.num_segments = args.num_segments
        self.modality = args.modality
        self.n_ways = args.train_n
        self.n_shots = args.train_k
        self.n_queries = args.n_queries
        self.n_episodes = args.n_train_runs if test_mode==False else args.n_test_runs


        
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()
        self.label_index  = {}
        for i in range(len(self.video_list)):
            if self.video_list[i].label not in self.label_index:      
                self.label_index[self.video_list[i].label]=[]
            self.label_index[self.video_list[i].label].append(self.video_list[i])
        self.classes = list(self.label_index.keys())
        
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)
        
        if not test_mode:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(224),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.CenterCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        
        self.test_transform = transforms.Compose([transforms.ToTensor(),normalize])

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        if self.fix_seed:
            np.random.seed(index)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        query_xs = []
        for idx, the_cls in enumerate(cls_sampled):
            vids = self.label_index[the_cls]
            support_xs_ids_sampled = np.random.choice(range(len(vids)), self.n_shots, False)
            for index in support_xs_ids_sampled:
                record = self.label_index[the_cls][index]
                segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                support_xs.extend([self.get(record, segment_indices)])
          
            query_xs_ids_sampled = np.random.choice(range(len(vids)), self.n_queries, False)
            for index in query_xs_ids_sampled:
                record = self.label_index[the_cls][index]
                segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                query_xs.extend([self.get(record, segment_indices)])
               
        #if not self.test_mode:
        #    segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        #else:
        #    segment_indices = self._get_test_indices(record)

        return support_xs,query_xs

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return self.n_episodes
