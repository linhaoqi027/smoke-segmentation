from torch.utils.data import Dataset
import numpy as np
import random
import os
import os
import glob
import numpy as np
import cv2
import random
import math

class TrainDataset(Dataset):
    def __init__(self):
        self.random_flip = True
        self.length = len(glob.glob('/home/ecust/lhq/smoke/training_data/blendall/*'))
        # this path is the data directory, you can change it for your need.
        self.root_path = '/home/ecust/lhq/smoke/training_data/'
        # assert args.mask, "Missing mask as the input"
        # assert args.normalization, "You need to do the data normalization before training"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pre=math.floor((idx-idx%7848)/7848)+1
        post=idx%7848+1
        im=cv2.cvtColor(cv2.imread(self.root_path+'blendall/'+ str(pre) + '_'+str(post) + '.jpg'), cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.root_path+'gt_blendall/'+ str(pre) + '_'+str(post) + '.png', cv2.IMREAD_GRAYSCALE)
        # im=cv2.cvtColor(cv2.imread(self.root_path+'blendall/'+ '1_'+str(idx+1) + '.jpg'), cv2.COLOR_BGR2RGB)
        # label = cv2.imread(self.root_path+'gt_blendall/'+ '1_'+str(idx+1) + '.png', cv2.IMREAD_GRAYSCALE)
        label = label/255
        im = im.transpose(2, 0, 1)
        # images shape: 3 x H x W
        # labels shape: H x W
        
        # if self.random_flip:
        #     flip_1 = np.random.choice(2) * 2 - 1#水平翻转
        #     im = im[:, ::flip_1, :]
        #     label = label[::flip_1, :]

        #     flip_2 = np.random.choice(2) * 2 - 1#上下翻转
        #     im = im[:,:,::flip_2]
        #     label = label[:,::flip_2]
        sample = {'images': im.copy(), 'labels': label.copy()}
        return sample


class TestDataset(Dataset):
    def __init__(self):
        #self.length = len(glob.glob('/home/ecust/lhq/smoke/8/*'))
        # this path is the data directory, you can change it for your need.
        self.root_path = '/home/ecust/lhq/0/'
        #self.root_path = '/home/ecust/lhq/smoke/testing_data/'
        # assert args.mask, "Missing mask as the input"
        # assert args.normalization, "You need to do the data normalization before training"

    def __len__(self):
        return 400


    def __getitem__(self, idx):
        # im = cv2.cvtColor(cv2.imread(self.root_path+'pic/'+str(idx+1) + '.png'),cv2.COLOR_BGR2RGB)
        # label = cv2.imread(self.root_path+'cv2_mask/'+str(idx+1) + '.png', cv2.IMREAD_GRAYSCALE)
        # im=cv2.resize(im, (256, 256))
        # label=cv2.resize(label, (256, 256))
        # label=np.where(label==0, 0,1)
        # im = im.transpose(2, 0, 1)
        im = cv2.cvtColor(cv2.imread(self.root_path+'pic/'+str(idx+1) + '.png'),cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.root_path+'cv2_mask/'+str(idx+1) + '.png', cv2.IMREAD_GRAYSCALE)
        label = np.where(label == 0, 0, 1)
        im = im.transpose(2, 0, 1)
        # images shape: 3 x H x W
        # labels shape: H x W
        sample = {'images': im, 'labels': label}
        return sample


