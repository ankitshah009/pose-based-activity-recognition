import argparse
import os
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

from potnet import * 
#from sed_dataloader import *
#from alexnet import *
from dataloader.potion_dataloader import *
# Initializing path constants
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_DIR = os.getcwd()

#DATA_DIR = '/home/aniru/repo/dataset/sed'
#VIDEO_LIST = DATA_DIR

#RGB_DIR_GT  = DATA_DIR + '/gt/i3d_rgb/'
#RGB_DIR_NEG  = DATA_DIR + '/negative/i3d_rgb/'

#POSE_DIR_GT = DATA_DIR + '/gt/openpose_heatmap.jiac/'
#POSE_DIR_NEG = DATA_DIR + '/negative/openpose_heatmap.jiac/'

#POTION_DIR_GT = DATA_DIR + '/potion/gt/'
#POTION_DIR_NEG = DATA_DIR + '/potion/negative/'

# frankenstien machine paths
DATA_DIR = '/media/hdd1/aps1/aniru/aniru/repo/dataset/UCF101'
RGB_DIR  = DATA_DIR + '/jpegs_256/'
POSE_DIR = DATA_DIR + '/heatmaps/'
UCF_LIST = BASE_DIR + '/UCF_list/'

#Initialize arguments
parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=25, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--classes', default=101, type=int, metavar='N', help='number of classes')
args = parser.parse_args()
# Create model, optimizer and loss function
model = alexnet(pretrained=False,num_classes=args.classes)
model.xavier_init()
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()


# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):
    lr = args.lr

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_models(epoch):
    torch.save(model.state_dict(), "PotionModel_{}.model".format(epoch))
    print("Chekcpoint saved")


def test(test_loader):
    model.eval()
    test_acc = 0.0
    
    nb_classes = args.classes
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    for i, (keys, data_dict, labels) in enumerate(test_loader):
        images = data_dict['potion']
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        #Swap the channel and height axes to get C,W,H
        #images = images.transpose(1,3).float()

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        # print("Prediction size:")
        # print(prediction.size())
        test_acc += torch.sum(prediction == labels.data)

        # Calculating AP for each class:
        for t, p in zip(labels.data.view(-1), prediction.view(-1)):
	    #print(t.shape,type(t),type(p))
            confusion_matrix[long(t), long(p)] += 1

    # print(confusion_matrix)
    print("Test-time classwise accuracy:")
    print(confusion_matrix.diag()/confusion_matrix.sum(1))

    # Compute the average acc and loss over all test images
    test_acc = float(test_acc.item())/ (len(test_loader)*args.batch_size)

    return test_acc


def train(num_epochs, train_loader, test_loader):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
	#### parameter to be changed
        nb_classes = args.classes
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        progress = tqdm(train_loader)

        for i, (data_dict,labels) in enumerate(progress):

            images = data_dict['potion']
	    #print("Here",images.shape)
            #Swap the channel and height to get C,W,H
            #images = images.transpose(1,3).float()

            # print(images.size())
            # print(images.size())

            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)
            
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            
            train_acc += torch.sum(prediction == labels.data)

            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print("PREDICTIONS:")
            # print(prediction)
            # print("LABELS:")
            # print(labels)
            # print(train_acc.item())
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            # Calculating AP for each class:
            for t, p in zip(labels.data.view(-1), prediction.view(-1)):
	    	#print(type(t),type(p))
                confusion_matrix[long(t), long(p)] += 1

        # print(confusion_matrix)
        print("Training classwise accuracy:")
        print(confusion_matrix.diag()/confusion_matrix.sum(1))

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        print("train_acc",train_acc.item())
        # Compute the average acc and loss over all training images
        train_acc = float(train_acc.item()) / (len(train_loader)*args.batch_size)
        train_loss = train_loss.item() / (len(train_loader)*args.batch_size)

        # Evaluate on the test set
        test_acc = test(test_loader)

        # print("RESULTS:")
        # print("###################################")
        # print(train_acc, len(train_loader), test_acc, len(test_loader), train_loss)
        # print("###################################")

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss, test_acc)) 


if __name__ == "__main__":

    global arg
    arg = parser.parse_args()
    print(arg)

    # Check if gpu support is available
    cuda_avail = torch.cuda.is_available()

    #if cuda is available, move the model to the GPU
    if cuda_avail:
        model.cuda()

    #Prepare DataLoader
    ## SED dataset
    #data_loader = sed_dataloader(
    #                    BATCH_SIZE=arg.batch_size,
    #                    num_workers=8,
    #                    rgb_path=DATA_DIR,
    #                    pose_path_gt=POSE_DIR_GT,
    #                    pose_path_neg=POSE_DIR_NEG
    #                    )
    data_loader = potion_dataloader(
			BATCH_SIZE=arg.batch_size,
			num_workers=19,
			rgb_path=DATA_DIR,
			pose_path=POSE_DIR,
			ucf_list=UCF_LIST,
			ucf_split='01'
			)

    train_loader, test_loader, test_video = data_loader.run()

    train(arg.epochs, train_loader, test_loader)
