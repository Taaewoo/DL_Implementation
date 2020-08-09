import os
import numpy as np
import torch
import random

from PIL import Image
from torch.utils.data import Dataset
from model import *


class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.triplet = self.generate_triplet()
        
    def __getitem__(self, idx):
        
        anc_path, pos_path, neg_path = self.triplet[idx]

        anc_img = Image.open(anc_path)
        pos_img = Image.open(pos_path)
        neg_img = Image.open(neg_path)
        
        anc_img = np.array(anc_img)
        pos_img = np.array(pos_img)
        neg_img = np.array(neg_img)
        
        sample = {
            'anc_img' : anc_img,
            'pos_img' : pos_img,
            'neg_img' : neg_img,
            'anc_path' : anc_path,
            'pos_path' : pos_path,
            'neg_path' : neg_path,
        }
        
        sample['anc_img'] = self.transform(sample['anc_img'])
        sample['pos_img'] = self.transform(sample['pos_img'])
        sample['neg_img'] = self.transform(sample['neg_img'])
        
        return sample
    
    def __len__(self):
        return len(self.triplet)
    
    def generate_triplet(self):
        
        classes = os.listdir(self.root_dir)
        
        #[ [class 1 이미지 path들] [class 2 이미지 path들] ... ]
        image_path = []

        # select random 40 people
        selected_classes = random.sample(classes, 40)
        for cls in selected_classes:
            #print("Class num : " + str(cls))
        
            class_image_path = []
        
            images = os.listdir(self.root_dir + "/" + cls)
            #print(min(len(images),40))

            # select random images per person
            selected_images = random.sample(images, min(len(images),40))
            for img in selected_images:
                class_image_path.append(self.root_dir + "/" + cls + "/" + img)
    
            #for path in class_image_path:
            #    print("    ", path)
    
            image_path.append(class_image_path)
        
        
        triplet = []
        
        for i in range(len(image_path)):
            
            for a_i in range(len(image_path[i])):
                for p_i in range(a_i+1,len(image_path[i])):
                    anc = image_path[i][a_i]
                    pos = image_path[i][p_i]
            
                    for j in range(len(image_path)):
                        if(i==j):
                            continue
                        
                        neg = image_path[j][0]
                    
                        triplet.append([anc, pos, neg])
                        #for n_i in range(len(image_path[j])):
                            #temp_triplet['Neg'] = n_i
                            
        print(len(triplet))
        random.shuffle(triplet)
        return triplet
