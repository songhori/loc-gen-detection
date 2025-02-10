import timm
import torch
import numpy as np
import pandas as pd
from torch import nn
from glob import glob
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from torchvision import models
from torchvision import transforms as T
# from insightface.app import FaceAnalysis
from deepface import DeepFace
tqdm.pandas()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(device)
# print("CUDA Available:", torch.cuda.is_available())
# print("cuDNN Version:", torch.backends.cudnn.version())



############ Initialize Parameters  ############

yolo_confidence = 0.7
body_model = 'models/yolo/yolo11x.pt'
gender_model_selection = 'vit_b_16'
location_model_name = 'vit_l_16'


match gender_model_selection:

    case 'EffNetb0Model' | 'EffNetV2sModel':
        gender_model = 'models/gender/gender_effb0_body.pth'
        img_size_gen = 224
        gender_threshold = 0.5

    case 'resnet152':
        gender_model = 'models/gender/resnet152_gender_fined.pth'
        img_size_gen = (224, 224)

    case 'efficientnet_b7_ns':
        gender_model = 'models/gender/efficientnet_b7_ns_gender_fined.pth'
        img_size_gen = (224, 224)

    case 'vit_b_16':
        gender_model = 'models/gender/vit_b_16_gender_fined_best.pth'
        img_size_gen = (384, 384)

    case 'vit_l_16':
        gender_model = 'models/gender/vit_l_16_gender_fined.pth'
        img_size_gen = (512, 512)

    case 'deepface' | 'insightface':
        pass


match location_model_name:
    
    case 'resnet50':
        location_model = 'models/location/resnet50_fined.pt'
        img_size_loc = 224

    case 'resnet101':
        location_model = 'models/location/resnet101_fined.pth'
        img_size_loc = (224, 224)

    case 'vit_b_16':
        location_model = 'models/location/vit_b_16_fined.pth'
        img_size_loc = (384, 384)

    case 'vit_l_16':
        location_model = 'models/location/vit_l_16_fined_best.pth'
        img_size_loc = (512, 512)

    case 'vit_h_14':
        location_model = 'models/location/vit_h_14_fined.pth'
        img_size_loc = (518, 518)



############ Preprocessing and Trasnforms ############

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform_gen = T.Compose([
    T.Resize(img_size_gen),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)
    ])

transform_loc = T.Compose([
    T.Resize(img_size_loc),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)
    ])



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


class EffNetV2sModel(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(EffNetV2sModel, self).__init__()
        self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x


class vitModel(nn.Module):
    def __init__(self, variant='b_16', num_classes=2, model_path=None):
        super(vitModel, self).__init__()
        
        # Initialize the vit model
        match variant:
            case 'b_16':
                self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            case 'l_16':
                self.model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            case 'h_14':
                self.model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
        
        # Replace the fully connected layer to match your fine-tuned model
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        
        # Load the saved state dict (if model_path is provided)
        if model_path:
            checkpoint = torch.load(model_path, map_location='cuda')
            self.model.load_state_dict(checkpoint)
        
    def forward(self, x):
        # Get raw logits from the model
        logits = self.model(x)
        
        # Apply softmax to convert logits to probabilities (along the class dimension)
        probabilities = torch.softmax(logits, dim=1)
        
        return probabilities


match gender_model_selection:

    case 'EffNetb0Model':
        gender = EffNetb0Model(num_classes=1)
        ckpt = torch.load(gender_model, map_location=device)
        gender.load_state_dict(ckpt)
        gender.to(device)
        gender.eval()

    case 'EffNetV2sModel':
        gender = EffNetV2sModel(num_classes=1)
        gender.to(device)
        gender.eval()

    case 'resnet152':
        gender = models.resnet152(weights=None)
        gender.fc = nn.Linear(gender.fc.in_features, 2)
        ckpt = torch.load(gender_model, map_location=device)
        gender.load_state_dict(ckpt)
        gender.to(device)
        gender.eval()

    case 'vit_b_16':
        gender = vitModel(num_classes=2, variant='b_16', model_path=gender_model)
        gender.to(device)
        gender.eval()
    
    case 'vit_l_16':
        gender = vitModel(num_classes=2, variant='l_16', model_path=gender_model)
        gender.to(device)
        gender.eval()

    case 'efficientnet_b7_ns':
        gender = timm.create_model('tf_efficientnet_b7.ns_jft_in1k', pretrained=False, num_classes=2)
        ckpt = torch.load(gender_model)
        gender.load_state_dict(ckpt)
        gender.to(device)
        gender.eval()

    case 'deepface':
        def get_deepface_gender(cropped_imm, deepface_detector = 'opencv', temp_path = "temp_cropped_image.jpg"):
            # Convert cropped image to a temporary file (DeepFace expects a file path)
            cropped_imm.save(temp_path)
            result = DeepFace.analyze(temp_path, actions=['gender'], detector_backend=deepface_detector, enforce_detection=False)
            gender = result[0]['dominant_gender']
            return gender

    case 'insightface':
        gender = FaceAnalysis(allowed_modules=['detection', 'genderage'])
        gender.prepare(ctx_id=0, det_size=(640, 640))  # Use GPU if available, or set ctx_id=-1 for CPU


print('gender model loaded')



############ Location Classification ############

class Resnet50Model(nn.Module):
    def __init__(self, num_classes=488):
        super(Resnet50Model, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.softmax(x, dim=1)
        return x
    

class ResNet101Model(nn.Module):
    def __init__(self, num_classes=488, pretrained=False, model_path=None):
        super(ResNet101Model, self).__init__()
        
        # Initialize the pre-trained ResNet101 model
        self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace the fully connected layer to match your fine-tuned model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Load the saved state dict (if model_path is provided)
        if model_path:
            checkpoint = torch.load(model_path, map_location='cuda')
            self.model.load_state_dict(checkpoint)
        
    def forward(self, x):
        """
        Forward pass through the network with softmax applied to the output.
        :param x: Input tensor
        :return: Probabilities for each class after softmax
        """
        # Get raw logits from the model
        logits = self.model(x)
        
        # Apply softmax to convert logits to probabilities (along the class dimension)
        probabilities = torch.softmax(logits, dim=1)
        
        return probabilities


match location_model_name:

    case 'resnet50':
        ckpt = torch.load(location_model, map_location=device)
        classifier = Resnet50Model(num_classes=488)
        classifier.load_state_dict(ckpt)

    case 'resnet101':
        classifier = ResNet101Model(num_classes=488, model_path=location_model)

    case 'vit_b_16':
        classifier = vitModel(num_classes=488, variant='b_16', model_path=location_model)

    case 'vit_l_16':
        classifier = vitModel(num_classes=488, variant='l_16', model_path=location_model)

    case 'vit_h_14':
        classifier = vitModel(num_classes=488, variant='h_14', model_path=location_model)


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
        
        match gender_model_selection:

            case 'EffNetb0Model' | 'EffNetV2sModel':
                imm_tra = transform_gen(imm)
                gen_pred = gender(imm_tra.unsqueeze(0).to(device)).detach().cpu().numpy()[0][0]
                if gen_pred < gender_threshold:
                    ma += 1
                else:
                    fe += 1
            
            case 'vit_b_16':
                imm_tra = transform_gen(imm)
                gen_pred = gender(imm_tra.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                if gen_pred[0] >= gen_pred[1]:
                    ma += 1
                else:
                    fe += 1

            case 'vit_l_16':
                imm_tra = transform_gen(imm)
                gen_pred = gender(imm_tra.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                if gen_pred[0] >= gen_pred[1]:
                    ma += 1
                else:
                    fe += 1

            case 'resnet152' | 'efficientnet_b7_ns':
                imm_tra = transform_gen(imm)
                tens_output = gender(imm_tra.unsqueeze(0).to(device))

                # Apply softmax to the tensor
                tens_probs = torch.softmax(tens_output, dim=1)

                # Convert to numpy and get predictions
                gen_pred = tens_probs.detach().cpu().numpy()[0]
                gen_pred_idx = np.argmax(gen_pred)

                # Update counters based on the prediction
                if gen_pred_idx == 0:
                    ma += 1
                else:
                    fe += 1

            case 'deepface':
                gender_pred = get_deepface_gender(imm)
                if gender_pred == 'Man':
                    ma += 1
                else:
                    fe += 1

            case 'insightface':
                faces = gender.get(np.array(imm))
                gen_pred = faces[0].sex  # 1 = Male, 0 = Female
                if gen_pred == 1:
                    ma += 1
                else:
                    fe += 1

    males.append(ma)
    females.append(fe)
    
    img = transform_loc(img)
    with torch.no_grad():
        # with torch.amp.autocast(device_type='cuda'):
        output = classifier(img.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
    probas.append(output)
    paths.append(filename.split('/')[-1])

sub = pd.DataFrame({'path':paths, 'males':males, 'females':females, 'probas':probas})



############ create submission file  ############

temp = sub.probas.apply(pd.Series)
sub = pd.concat([sub,temp], axis=1)
sub = sub.drop(['probas'], axis=1)

sub.to_csv('submission/submission.csv', index=False)
