import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
import random
# from skimage import io, color, exposure
import cv2
import os
import numpy as np
import nonechucks as nc
import torch
from pathlib import Path

BASE_DIR = os.getcwd()
DATA_DIR = '/data/MM1/aps1/aniru/aniru/repo/dataset/sed'
# DATA_DIR = '/home/ubuntu/dataset/UCF101'

VIDEO_LIST = DATA_DIR

RGB_DIR_GT  = DATA_DIR + '/gt/i3d_rgb/'
RGB_DIR_NEG  = DATA_DIR + '/negative/i3d_rgb/'

POSE_DIR_GT = DATA_DIR + '/gt/openpose_heatmap.jiac/'
POSE_DIR_NEG = DATA_DIR + '/negative/openpose_heatmap.jiac/'

POTION_DIR_GT = DATA_DIR + '/potion/gt/'
POTION_DIR_NEG = DATA_DIR + '/potion/negative/'

# # Sample structure
# {'event_begin': 117570, 'start_frame': 117554, 'event_end': 117586, 
#     'props_label': 'CellToEar', 'props_type': 'gt', 'end_frame': 117602, 'props_id': 1, 
#     'props_name': 'LGW_20071106_E1_CAM3_1', 'video_name': 'LGW_20071106_E1_CAM3'}

class sed_dataset(Dataset):  
    def __init__(self, dic, rgb_dir, pose_dir_gt, pose_dir_neg, mode, transform=None):
 
        # self.keys = list(dic.keys())
        # self.values= list(dic.values())

        self.dic = dic
        self.rgb_dir = rgb_dir
        self.pose_dir_gt = pose_dir_gt
        self.pose_dir_neg = pose_dir_neg
        self.mode =mode
        self.transform = transform
        self.action_label = self.get_action_index()

    def __len__(self):
        return len(self.dic)

    def potion_transform(self, video_name, video_type):

        if video_type == 'gt':
            path = self.pose_dir_gt + video_name + '.npz'
            potion_dir = POTION_DIR_GT + video_name + "_potion.npy"
        else:
            path = self.pose_dir_neg + video_name + '.npz'
            potion_dir = POTION_DIR_NEG + video_name + "_potion.npy"

        try:
            pot = np.load(potion_dir)
            # print("Found existing potion feature. Loading....", pot.shape)
        except:
            pot = None

        if pot is not None:
            return pot
        else:
            try:
                tracklet = np.load(path)
            except:
                raise FileNotFoundError('FileDoesNotExist')
                # return None
            
            #Initialize flags for color control
            b = False
            r = g = True
            canvas_split = len(tracklet['feat'])//2 + 1

            for idx, poseframe in enumerate(tracklet['feat']):
                # Modulate ratio to get blend of red-green and green-blue
                ratio = (idx % canvas_split)/canvas_split
                if idx >= canvas_split:
                    r = False
                    b = True
                    ratio = 1 - ratio

                colorizer = np.zeros((poseframe.shape[0], poseframe.shape[1], 3))
                colorizer[np.where((colorizer == [0, 0, 0]).all(axis=2))] = \
                                    [b*(1 - ratio)*255/canvas_split, g*ratio*255/canvas_split, r*(1-ratio)*255/canvas_split]
                
                potion_t = np.concatenate((poseframe, colorizer), axis=2)

                if idx == 0:
                    potion = potion_t
                else:
                    potion = cv2.add(potion, potion_t)
                  
            # Save PoTion image
            np.save(potion_dir, potion)

            return potion

    def get_action_index(self):
        action_label={}
        with open(BASE_DIR +'/sedClassIdx.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label,action = line.split(' ')

            if action not in action_label.keys():
                action_label[action]=label
        return action_label

    def load_ucf_image(self, video_name, video_type):
        # rgb_path = self.rgb_dir + 'v_' + video_name #+'/v_'+video_name+'_'
  
        # PoTion transformation
        img = self.potion_transform(video_name, video_type)

        if img is None:
            return None
        else:
            # img = Image.fromarray(img)
            # transformed_img = self.transform(img)
            # img.close()
            transformed_img = img

            return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train' or self.mode == 'val':
            video_name = self.dic[idx]['props_name']
            video_type = self.dic[idx]['props_type']
        else:
            raise ValueError('There are only train and val mode')

        # label = self.values[idx]
        # label = self.dic[idx]['props_label']
        label = self.action_label[self.dic[idx]['props_label']]
        label = int(label)-1

        # Maintaining a dict for future extension of features        
        data ={}
        potion_f = self.load_ucf_image(video_name, video_type)
        
        # if potion_f is None:
        #     data['potion'] = np.zeros([20, 20, 60])
        # else:
        data['potion'] = potion_f

        if self.mode=='train':
            sample = (data, label)
        elif self.mode=='val':
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class sed_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, rgb_path, pose_path_gt, pose_path_neg):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.rgb_path=rgb_path
        self.pose_path_gt=pose_path_gt
        self.pose_path_neg=pose_path_neg
        self.frame_count={}

        # load the action labels & training and testing videos
        self.train_video = self.load_train()
        self.test_video = self.load_test()

    def load_train(self):
        with open(DATA_DIR + '/train.pkl', "rb") as file:
            dic = pickle.load(file)

        train_dic = []
        for sample in dic:
            if sample['props_type'] == 'gt':
                path = POSE_DIR_GT + sample['props_name'] + '.npz'
            else:
                path = POSE_DIR_NEG + sample['props_name'] + '.npz'

            my_file = Path(path)
            if my_file.is_file():
                train_dic.append(sample)

        # print("TRAIN DIC: ", len(dic), len(train_dic))
        return train_dic

    def load_test(self):
        with open(DATA_DIR + '/test.pkl', "rb") as file:
            dic = pickle.load(file)

        test_dic = []
        for sample in dic:
            if sample['props_type'] == 'gt':
                path = POSE_DIR_GT + sample['props_name'] + '.npz'
            else:
                path = POSE_DIR_NEG + sample['props_name'] + '.npz'

            my_file = Path(path)
            if my_file.is_file():
                test_dic.append(sample)
            
        # print("TEST DIC: ", len(dic), len(test_dic))
        return test_dic

    def run(self):
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def my_collate(self, batch):
        batch = list(filter(lambda x:x is not None, batch))
        return default_collate(batch)

    def train(self):
        training_set = sed_dataset(dic=self.train_video, rgb_dir=self.rgb_path, pose_dir_gt=self.pose_path_gt, pose_dir_neg=self.pose_path_neg, \
                mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        # training_set = nc.SafeDataset(training_set)
        print('==> Training data :',len(training_set),'frames')

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            # collate_fn=self.my_collate,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = sed_dataset(dic=self.test_video, rgb_dir=self.rgb_path, pose_dir_gt=self.pose_path_gt, pose_dir_neg=self.pose_path_neg, \
                mode='val', transform = transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        # validation_set = nc.SafeDataset(validation_set)
        print('==> Validation data :',len(validation_set),'frames')

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            # collate_fn=self.my_collate,
            num_workers=self.num_workers)
        return val_loader

if __name__ == '__main__':
    
    dataloader = sed_dataloader(BATCH_SIZE=1, num_workers=1, 
                                rgb_path=DATA_DIR,
                                pose_path_gt=POSE_DIR_GT,
                                pose_path_neg=POSE_DIR_NEG
                                )
    train_loader,val_loader,test_video = dataloader.run()
