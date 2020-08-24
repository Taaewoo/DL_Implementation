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
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.distance import PairwiseDistance
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from validate_on_LFW import evaluate_lfw
from models.resnet_triplet import Resnet50Triplet
from models.resnet_triplet import Resnet34Triplet
from models.efficientnet import *

num_epoch = 500
num_batch_per_epoch = 100
people_per_batch = 45
images_per_person = 40
margin = 0.5
random_seed = 42
batch_size = 32

writer = SummaryWriter('tensorboard/Efficient2/')

model = EfficientnetTriplet()

flag_train_gpu = torch.cuda.is_available()

if flag_train_gpu:
    model.cuda()
    print("Using single-gpu training")
else:
    print("No GPU!!!")
    
trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(224),
                            transforms.ToTensor(),
                           transforms.Normalize(
                                (0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))
                           ])


# 1 epoch 마다 embedding 하고 triplet selection.
# 1 epoch에선 40명 40장씩 샘플을 뽑고 적당한 triplet을 구성하고 학습했을 때
# 1 epoch의 batch_size 만큼 안채워졌으면 다시 40명 40장을 뽑음.

a_p_distance = []
a_n_distance = []
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
l2_distance = PairwiseDistance(2).cuda()

for ep in range(num_epoch):
    
    train_dataset = TripletDataset(root_dir="data/CASIA-WebFace-MTCNN",
                             transform = trans)

    val_dataset = TripletDataset(root_dir="data/CASIA-WebFace-MTCNN-Val",
                             transform = trans)

    # trainset은 길이가 정해져 있고 한번 loop에 batch_size만큼 가져옴.
    trainloader = DataLoader(dataset=train_dataset, batch_size = batch_size,
                             num_workers=35)
    val_loader = DataLoader(dataset=val_dataset, batch_size = batch_size,
                             num_workers=35)
    
    loss = 0
    batch_cnt = 0
    triplet_size = 0
    for i, data in enumerate(trainloader):
        print("Epoch :",ep+1,"(",i+1,"/",num_batch_per_epoch,")", end='')

        #f, axarr = plt.subplots(1,3)

        for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()


        anc_fv = model(data['anc_img'].cuda())
        pos_fv = model(data['pos_img'].cuda())
        neg_fv = model(data['neg_img'].cuda())

        #print("Batch Size :",anc_fv.shape[0])


        pos_dis = l2_distance.forward(anc_fv,pos_fv)
        neg_dis = l2_distance.forward(anc_fv,neg_fv)

        all = (pos_dis < neg_dis).cpu().numpy().flatten()

        losses = torch.clamp(pos_dis[all] - neg_dis[all] + margin, min=0.0)
        
        loss = torch.mean(losses)
            
        print(" Len :", len(pos_dis[all]), end='')
        print(" Loss :",loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        if i+1 == num_batch_per_epoch:
            break

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            label = np.zeros(batch_size)
            label[:len(label)//2] = 1
            np.random.shuffle(label)

            pos_idx = (label == 1)
            neg_idx = (label == 0)

            anc_img_p = data['anc_img'][pos_idx]
            pos_img = data['pos_img'][pos_idx]

            anc_img_n = data['anc_img'][neg_idx]
            neg_img = data['neg_img'][neg_idx]

            '''
            f, axarr = plt.subplots(1,2)

            if(label[0]==0):
                print("*Same")
                print("Anc Path :", data['anc_path'][0])
                print("Pos Path :", data['pos_path'][0])
                axarr[0].imshow(anc_img_p[0].permute(1,2,0))
                axarr[1].imshow(pos_img[0].permute(1,2,0))
            else:
                print("*Different")
                print("Anc Path :", data['anc_path'][0])
                print("Neg Path :", data['neg_path'][0])
                axarr[0].imshow(anc_img_n[0].permute(1,2,0))
                axarr[1].imshow(neg_img[0].permute(1,2,0))

            plt.show()
            '''

            anc_fv = model(data['anc_img'].cuda())
            pos_fv = model(data['pos_img'].cuda())
            neg_fv = model(data['neg_img'].cuda())

            pos_dis = l2_distance.forward(anc_fv,pos_fv).cpu().detach().numpy()
            neg_dis = l2_distance.forward(anc_fv,neg_fv).cpu().detach().numpy()

            distance = np.where(label==1, pos_dis, neg_dis)

            true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
            tar, far = evaluate_lfw(
                distances=distance,
                labels=label
            )
            # Print statistics and add to log
            print("Accuracy on LFW: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
                  "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t"
                  "TAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
                    np.mean(accuracy),
                    np.std(accuracy),
                    np.mean(precision),
                    np.std(precision),
                    np.mean(recall),
                    np.std(recall),
                    roc_auc,
                    np.mean(best_distances),
                    np.std(best_distances),
                    np.mean(tar),
                    np.std(tar),
                    np.mean(far)
                  )
            )
            print("")

            with open('log_triplet.txt', 'a') as f:
                val_list = [
                    ep + 1,
                    np.mean(accuracy),
                    np.std(accuracy),
                    np.mean(precision),
                    np.std(precision),
                    np.mean(recall),
                    np.std(recall),
                    roc_auc,
                    np.mean(best_distances),
                    np.std(best_distances),
                    np.mean(tar)
                ]
                log = '\t'.join(str(value) for value in val_list)
                f.writelines(log + '\n')

            writer.add_scalar('Val Accuracy', np.mean(accuracy), ep+1)

            break
        
    # Save model checkpoint
    if((ep+1)%10 == 0):
        torch.save(state, 'save_models/Efficient2/epoch_{}.pt'.format(ep + 1))
            
       
    

    




