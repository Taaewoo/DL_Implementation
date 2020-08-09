import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import torchvision

from model import *
from triplet_dataloader import *
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

num_epoch = 500
num_batch_per_epoch = 1000
people_per_batch = 45
images_per_person = 40
margin = 0.2

def cal_l2_distance(a, b):
    return ((a - b)**2).sum().item()

model = FaceNet()

trans = transforms.Compose([transforms.ToTensor()])

# trainset은 길이가 정해져 있고 한번 loop에 batch_size만큼 가져옴.
trainloader = DataLoader(dataset=TripletDataset(root_dir="data/CASIA-WebFace-MTCNN", transform = trans), batch_size = 64, num_workers=20)


a_p_distance = []
a_n_distance = []

for i, data in enumerate(trainloader):
    print("Batch Number :", i+1)
    
    anc_fv = model(data['anc_img'])
    pos_fv = model(data['pos_img'])
    neg_fv = model(data['neg_img'])
    
    print("Batch Size :",anc_fv.shape[0])
    
    for j in range(anc_fv.shape[0]):
        pos_dis = cal_l2_distance(anc_fv[j],pos_fv[j])
        neg_dis = cal_l2_distance(anc_fv[j],neg_fv[j])
        
        print("Anchor <-> Positive :", pos_dis)
        print("Anchor <-> Negative :", neg_dis)
        
        if(pos_dis < neg_dis):
            print("Semi - hard")
            a_p_distance.append(pos_dis)
            a_n_distance.append(neg_dis)
            
    print("")
    
    if(len(a_p_distance) > 64):
        a_p_distance = torch.tensor(a_p_distance)
        a_n_distance = torch.tensor(a_n_distance)
        
        losses = torch.clamp(a_p_distance - a_n_distance + margin, min=0.0)
        loss = torch.mean(losses)
        print(losses)
        print(loss)
        break
