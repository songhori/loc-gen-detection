############ Load libraries  ############

import numpy as np
import sys
from PIL import Image
import torch
from glob import glob
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import transforms, datasets, models
tqdm.pandas()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda'



############ Preprocessing and Trasnforms ############

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = T.Compose([T.Resize(224),
                       T.ToTensor(),
                       T.Normalize(*imagenet_stats)])



############ Gender Classification ############

class EffModel(torch.nn.Module):
    def __init__(self,num_classes=1000):
        super(EffModel, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
        
    def forward(self, x):
        x = self.model(x)
        x = F.sigmoid(x)
        return x


ckpt = torch.load('gender_effb0_body.pth', map_location=device)
gender = EffModel(num_classes=1)
gender.load_state_dict(ckpt)
gender.to(device)
gender.eval()

print('gender model loaded')



############ Location Classification ############

class ResnetModel(nn.Module):
    def __init__(self,num_classes=1000):
        super(ResnetModel, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.softmax(x)
        return x
    
ckpt = torch.load('resnet50_fined.pt', map_location=device)
classifier = ResnetModel(num_classes=488)
classifier.load_state_dict(ckpt)
classifier.eval()
classifier.to(device)

print('loc classifier model loaded')



############ Body Detection ############

yolo_model = YOLO('yolov8n.pt',)
yolo_model.to(device)
dict_classes = yolo_model.model.names
class_IDS = [0]

print('detection model loaded')



############ Body Detection, Gender Classification, Location Classification ############

probas = []
males = []
females = []
paths = []

test_path = '/media/nextera/New Volume/facecup/Facecup/test/*'
files = glob(test_path)
for filename in tqdm(files):
    cnt = 0
    ma  = 0
    fe  = 0
    
    img = Image.open(filename).convert('RGB')

    out = yolo_model.predict(img,verbose=False ,conf=0.2,classes =[0])
    for bb in out[0].boxes.xyxy:
        cnt += 1
        bb = bb.detach().cpu().numpy().astype(int)
        imm = img.crop([bb[0],bb[1],bb[2],bb[3]])
        
        imm = transform(imm)
        gen_pred = gender(imm.unsqueeze(0).to(device)).detach().cpu().numpy()[0][0]
        
        if gen_pred<0.4:
            ma+=1
        else:
            fe+=1
            
    males.append(ma)
    females.append(fe)
    img = transform(img)
    probas.append(classifier(img.unsqueeze(0).to(device)).detach().cpu().numpy()[0])
    paths.append(filename.split('/')[-1])


sub = pd.DataFrame({'path':paths,'males':males,'females':females,'probas':probas})



############ create submission file  ############

temp = sub.probas.apply(pd.Series)
sub = pd.concat([sub,temp],axis=1)
sub = sub.drop(['probas'],axis=1)

sub.to_csv('submission.csv',index=False)

sub
