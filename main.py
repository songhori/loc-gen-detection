############ Load libraries  ############

from PIL import Image
import torch
from glob import glob
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
from torch import nn
from torchvision import transforms as T
from torchvision import models
tqdm.pandas()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(device)
# print("CUDA Available:", torch.cuda.is_available())
# print("cuDNN Version:", torch.backends.cudnn.version())



############ Initialize Parameters  ############

yolo_confidence = 0.6
body_model = 'models/yolo/yolo11x.pt'

gender_threshold = 0.5
gender_model = 'models/gender/gender_effb0_body.pth'

location_model = 'models/resnet/resnet50_fined.pt'



############ Preprocessing and Trasnforms ############

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = T.Compose([T.Resize(224),
                       T.ToTensor(),
                       T.Normalize(*imagenet_stats)])



############ Body Detection ############

yolo_model = YOLO(body_model)
yolo_model.to(device)
dict_classes = yolo_model.model.names
class_IDS = [0]

print('body detection model loaded')



############ Gender Classification ############

class EffNetb0Model(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(EffNetb0Model, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x

gender = EffNetb0Model(num_classes=1)
ckpt = torch.load(gender_model, map_location=device)
gender.load_state_dict(ckpt)


# class EffNetV2sModel(torch.nn.Module):
#     def __init__(self, num_classes=1000):
#         super(EffNetV2sModel, self).__init__()
#         self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
#         self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

#     def forward(self, x):
#         x = self.model(x)
#         x = torch.sigmoid(x)
#         return x

# gender = EffNetV2sModel(num_classes=1)


gender.to(device)
gender.eval()

print('gender model loaded')



############ Location Classification ############

class ResnetModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResnetModel, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.softmax(x, dim=1)
        return x
    
ckpt = torch.load(location_model, map_location=device)
classifier = ResnetModel(num_classes=488)
classifier.load_state_dict(ckpt)
classifier.eval()
classifier.to(device)

print('location classifier model loaded')



############ Body Detection, Gender Classification, Location Classification ############

probas = []
males = []
females = []
paths = []

test_path = 'data/test/*'
files = glob(test_path)
for filename in tqdm(files):
    cnt = 0
    ma  = 0
    fe  = 0
    
    img = Image.open(filename).convert('RGB')
    out = yolo_model.predict(img, verbose=False, conf=yolo_confidence, classes=[0], device=device)

    for bb in out[0].boxes.xyxy:
        cnt += 1
        bb = bb.detach().cpu().numpy().astype(int)
        imm = img.crop([bb[0], bb[1], bb[2], bb[3]])
        
        imm = transform(imm)
        gen_pred = gender(imm.unsqueeze(0).to(device)).detach().cpu().numpy()[0][0]
        
        if gen_pred < gender_threshold:
            ma+=1
        else:
            fe+=1
            
    males.append(ma)
    females.append(fe)
    img = transform(img)
    probas.append(classifier(img.unsqueeze(0).to(device)).detach().cpu().numpy()[0])
    paths.append(filename.split('/')[-1])

sub = pd.DataFrame({'path':paths, 'males':males, 'females':females, 'probas':probas})



############ create submission file  ############

temp = sub.probas.apply(pd.Series)
sub = pd.concat([sub,temp], axis=1)
sub = sub.drop(['probas'], axis=1)

sub.to_csv('submission.csv',index=False)
