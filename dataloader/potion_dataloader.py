import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from .split_train_test_video import *
from skimage import io, color, exposure
import cv2
import os
import numpy as np

BASE_DIR = os.getcwd()
#DATA_DIR = '/media/bighdd1/arayasam/dataset/UCF101'
# DATA_DIR = '/home/ubuntu/dataset/UCF101'
DATA_DIR = '/data/MM1/aps1/aniru/aniru/repo/dataset/UCF101'
RGB_DIR  = DATA_DIR + '/jpegs_256/'
POSE_DIR = DATA_DIR + '/heatmaps/'
POTION_DIR = DATA_DIR + '/potion/'
UCF_LIST = BASE_DIR + '/UCF_list/'


class potion_dataset(Dataset):  
    def __init__(self, dic, rgb_dir, pose_dir, mode, transform=None):
 
        self.keys = list(dic.keys())
        self.values= list(dic.values())
        self.rgb_dir = rgb_dir
        self.pose_dir = pose_dir
        self.mode =mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def potion_transform(self, video_name):
        path = self.pose_dir + 'v_' + video_name + '/'
        potion_dir = POTION_DIR + video_name + "_agg_image.png" 
        pot = cv2.imread(potion_dir)
        if pot is not None:
            return pot
        else:
            images = []
            for filename in sorted(os.listdir(path)):
                img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
                # Transform image to monochrome
                if img is not None:
                    potion_t = img.copy()
                    potion_t[np.where(potion_t > [0])] = [255]
                    potion_t = cv2.cvtColor(potion_t, cv2.COLOR_GRAY2BGR)
                    images.append(potion_t)

            b = False
            r = g = True
            pose_list = []
            agg_image = images[0]
            agg_image[np.where(True)] = [0, 0, 0]
            canvas_split = len(images)//2 + 1
            for idx, img in enumerate(images):
                # Modulate ratio to get blend of red-green and green-blue
                ratio = (idx % canvas_split)/canvas_split
                if idx >= canvas_split:
                    r = False
                    b = True
                    ratio = 1 - ratio

                # Normalize the pixel intensities - no. of frames
                img[np.where((img == [255, 255, 255]).all(axis=2))] = [b*(1 - ratio)*255/canvas_split, g*ratio*255/canvas_split, r*(1-ratio)*255/canvas_split]
                # img[np.where((img == [255, 255, 255]).all(axis=2))] = [b*(1 - ratio)*255, g*ratio*255, r*(1-ratio)*255]
                pose_list.append(img)

                agg_image = cv2.add(agg_image, img)
                
            # Save PoTion image
            cv2.imwrite(potion_dir, agg_image)

            # return images, pose_list, agg_image
            return agg_image

    def load_ucf_image(self, video_name):
        rgb_path = self.rgb_dir + 'v_' + video_name #+'/v_'+video_name+'_'
  
        # PoTion transformation
        # image_list, pose_list, img = self.potion_transform(video_name)
        # img = Image.open(rgb_path + '/frame000001.jpg')
        img = self.potion_transform(video_name)

        # Test image being loaded
        # print("Loading image validation:")
        # img.save("./load_ucf/" + video_name + ".png")

        img = Image.fromarray(img)
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name = self.keys[idx]
            
        elif self.mode == 'val':
            video_name = self.keys[idx]
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1
        
        data ={}
        if self.mode=='train':
            # Maintaining a dict for future extension of features
            data['potion'] = self.load_ucf_image(video_name)
                    
            sample = (data, label)
        elif self.mode=='val':
            data['potion'] = self.load_ucf_image(video_name)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class potion_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, rgb_path, pose_path, ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.rgb_path=rgb_path
        self.pose_path=pose_path
        self.frame_count={}
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video()

    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        with open(BASE_DIR + '/dataloader/dic/frame_count.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame :
            videoname = line.split('_',1)[1].split('.',1)[0]
            n,g = videoname.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            self.frame_count[videoname]=dic_frame[line]

    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        self.dic_training={}

        for video in self.train_video:
            nb_frame = self.frame_count[video]-10+1
            key = video+' '+ str(nb_frame)
            self.dic_training[key] = self.train_video[video]
                    
    def val_sample20(self):
        print('==> sampling testing frames')
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video]-10+1
            interval = int(nb_frame/19)
            for i in range(19):
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]

    def run(self):
        # self.load_frame_count()
        # self.get_training_dic()
        # self.val_sample20()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def train(self):
        training_set = potion_dataset(dic=self.train_video, rgb_dir=self.rgb_path, pose_dir=self.pose_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
		#transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print('==> Training data :',len(training_set),'frames')
        #print(training_set[1][1]['potion'].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = potion_dataset(dic=self.test_video, rgb_dir=self.rgb_path, pose_dir=self.pose_path, mode='val', transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print('==> Validation data :',len(validation_set),'frames')
        print(validation_set[1][1]['potion'].size())

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader

if __name__ == '__main__':
    
    dataloader = potion_dataloader(BATCH_SIZE=1, num_workers=1, 
                                rgb_path=RGB_DIR,
                                pose_path=POSE_DIR,
                                ucf_list=UCF_LIST,
                                ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()
