import os, json, cv2, numpy as np, matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
import albumentations as A # Library for augmentations
import numpy as np
from Datahelp import *
import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate
from openpyxl import Workbook
from torch.utils.tensorboard import SummaryWriter
import random




class ClassDataset(Dataset):
    def __init__(self,imgroot,annroot,demo=False):
        self.imgroot = imgroot
        self.annroot = annroot
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        imgs_file=os.listdir(imgroot)
        imgs_file.sort()
        imgs_file.sort(key=len)
        ann_file=os.listdir(annroot)
        ann_file.sort()
        ann_file.sort(key=len)
        self.imgs_files = imgs_file
        self.annotations_files = ann_file
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.imgroot,self.imgs_files[idx])
        annotations_path = os.path.join(self.annroot,self.annotations_files[idx])
        img_original = Image.open(img_path).convert('RGB')
        with open(annotations_path) as f:
            data_all = json.load(f)
            data=data_all['annotations']
            bboxes_original = data[0]['bbox']
            keypoints_original = data[0]['keypoints']
        img, bboxes, keypoints = img_original, bboxes_original, keypoints_original     


        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are glue tubes
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)        
        img = F.to_tensor(img)


        return img, target

    def __len__(self):
        return len(self.imgs_files)



imgroot='/home/p86091139/project/單手-5keypoint/image'
annroot='/home/p86091139/project/單手-5keypoint/ann'



def get_model(num_keypoints, weights_path=None):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


dataset = ClassDataset(imgroot=imgroot,annroot=annroot, demo=False)

train_sampler, val_sampler = load_split_train_test(dataset)

data_loader_train = DataLoader(dataset, batch_size=8, shuffle =False,sampler=train_sampler,collate_fn=collate_fn)
data_loader_test=DataLoader(dataset, batch_size=1, shuffle =False,sampler=val_sampler,collate_fn=collate_fn)
writer = SummaryWriter('/home/p86091139/project/loss')
def train():


    model = get_model(num_keypoints = 5)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    num_epochs = 300

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch,1000,writer)


    # Save model weights after training
    torch.save(model.state_dict(), '5keypointsrcnn3.pt')





def predict():
    train_transforms=transforms.ToTensor()
    model = get_model(num_keypoints = 5,weights_path="5keypointsrcnn.pt")


    model.to(device)



    with torch.no_grad():
        for i,x in enumerate(data_loader_test):
            images, targets=x
            image_id=targets[0]['image_id'].cpu().numpy()
            old_keypoints = []
            for kps in targets[0]['keypoints'].detach().cpu().numpy():
                old_keypoints.append([list(map(int, kp[:2])) for kp in kps])
            model.eval()
            images=list(image.to(device) for image in images)
            output = model(images)
            image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            scores = output[0]['scores'].detach().cpu().numpy()
            high_scores_idxs = np.where(scores > 0.8)[0].tolist() # Indexes of boxes with scores > 0.7
            post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
            keypoints = []
            for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])
            # bboxes = []
            # for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            #     bboxes.append(list(map(int, bbox.tolist())))
            visualize(image,keypoints=old_keypoints,path='5Keypoint-predict2',keypoints2=keypoints,num=i,name='combine_')
            data_save(num=i,img_id=image_id,old_keypoints=old_keypoints,keypoints=keypoints,path='5Keypoint-predict2',mood='txt')
    valuemean('5Keypoint-predict2')


def predict2():
    train_transforms=transforms.ToTensor()
    model = get_model(num_keypoints = 5,weights_path="5keypointsrcnn.pt")
    test_dataset = TestDataset(img_root='/home/p86091139/project/CLAHE',transform=train_transforms)
    dataloaders = DataLoader(dataset=test_dataset, batch_size=1,shuffle=False, collate_fn=collate_fn)
    # Visualizing model predictions

    model.to(device)

    with torch.no_grad():
        for i,x in enumerate(dataloaders):
            images, targets=x
            model.eval()
            images=list(image.to(device) for image in images)
            output = model(images)
            image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            scores = output[0]['scores'].detach().cpu().numpy()
            high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
            post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
            keypoints = []
            for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps]) 
            visualize(image,keypoints=keypoints,path='CLAHE_predict',num=i,name='predict_')
            data_save(num=i,img_id=i,old_keypoints=keypoints,path='CLAHE_predict',mood='txt')



print("-----------------")
print("開始訓練")
train()
print("-----------------")
print("開始測試")
predict()
print("-----------------")
print("結束訓練")








