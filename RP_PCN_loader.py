#from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import json
import numpy as np
import sys
import utils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path=os.path.abspath(os.path.join(BASE_DIR, '../dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/'))

class PartDataset(data.Dataset):
    def __init__(self, root=dataset_path, npoints=2500, classification=False, class_choice=None, split='train', normalize=False, crop_point_num=512):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.classification = classification
        self.normalize = normalize
        self.crop_point_num = crop_point_num

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
            print(self.cat)
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_point_fake = os.path.join(self.root, self.cat[item], 'point_fake')
            # print(dir_point)
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                sys.exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_point_fake, token + '.pts'), self.cat[item], token))            
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2], fn[3]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        # print(self.classes)
        self.cache = {}  # from index to (point_set, cls) tuple
        self.cache_size = 18000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, point_fake_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            point_fake_set = np.loadtxt(fn[2]).astype(np.float32)
            if self.normalize:
                point_set = self.pc_normalize(point_set)
                point_fake_set = self.pc_normalize(point_fake_set)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, point_fake_set, cls)

        choice1 = utils.iterative_farthest_point_sampling_torch(point_set, self.npoints)
        choice2 = utils.iterative_farthest_point_sampling_torch(point_fake_set, self.crop_point_num)
        point_set = point_set[choice1, :]
        point_fake_set = point_fake_set[choice2, :]
        point_set = torch.from_numpy(point_set)
        point_fake_set = torch.from_numpy(point_fake_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        
        return point_set, point_fake_set, cls

    def __len__(self):
        return len(self.datapath)
       
    def pc_normalize(self, pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

if __name__ == '__main__':
    dset = PartDataset( root='../dataset/mydata',classification=True, class_choice=None, npoints=4096, split='train')
#    d = PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=False, class_choice=None, npoints=4096, split='test')
    print(len(dset))
    ps, cls = dset[10]
    print(cls)
    print(ps.size(), ps.type(), cls.size(), cls.type())
    print(ps)
#    ps = ps.numpy()
#    np.savetxt('ps'+'.txt', ps, fmt = "%f %f %f")
