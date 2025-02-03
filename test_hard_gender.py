############ Load libraries  ############

import os
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
from sklearn.metrics import log_loss, accuracy_score
tqdm.pandas()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



############ Initialize Parameters  ############

test_path = 'data/bodies/test_hard_gender/*'
yolo_confidence = 0.7
body_model = 'models/yolo/yolo11x.pt'
gender_model_selection = 'vit_b_16'


match gender_model_selection:

    case 'EffNetb0Model' | 'EffNetV2sModel':
        gender_threshold = 0.5
        gender_model = 'models/gender/gender_effb0_body.pth'

    case 'resnet152':
        gender_model = 'models/gender/resnet152_gender_fined.pth'

    case 'vit_b_16':
        gender_model = 'models/gender/vit_b_16_gender_fined.pth'

    case 'vit_l_16':
        gender_model = 'models/gender/vit_l_16_gender_fined.pth'

    case 'efficientnet_b7_ns':
        gender_model = 'models/gender/efficientnet_b7_ns_gender_fined.pth'

    case 'deepface' | 'insightface':
        pass



############ Preprocessing and Trasnforms ############

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform = T.Compose([T.Resize(224),
                       T.ToTensor(),
                       T.Normalize(*imagenet_stats)])

transform2 = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)  # Add normalization
])

transform3 = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)  # Add normalization
])

def resize_with_padding(image, target_size=(224, 224)):
    # Resize while maintaining aspect ratio
    image = T.Resize(target_size, interpolation=Image.BILINEAR)(image)
    
    # Get new dimensions
    width, height = image.size
    
    # Calculate padding
    pad_height = (target_size[1] - height) // 2
    pad_width = (target_size[0] - width) // 2
    
    # Pad the image
    image = T.Pad((pad_width, pad_height, target_size[0] - width - pad_width, target_size[1] - height - pad_height))(image)
    return image

transform4 = T.Compose([
    T.Lambda(lambda img: resize_with_padding(img)),
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
    def __init__(self, variant='b_16', num_classes=2, pretrained=False, model_path=None):
        super(vitModel, self).__init__()
        
        # Initialize the vit model
        match variant:
            case 'b_16':
                self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            case 'l_16':
                self.model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace the fully connected layer to match your fine-tuned model
        # self.model.heads = nn.Linear(self.model.heads[0].in_features, num_classes)
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



############ Body Detection, Gender Classification, Location Classification ############

males = []
females = []
paths = []

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
                imm_tra = transform(imm)
                gen_pred = gender(imm_tra.unsqueeze(0).to(device)).detach().cpu().numpy()[0][0]
                if gen_pred < gender_threshold:
                    ma += 1
                else:
                    fe += 1
            
            case 'vit_b_16':
                imm_tra = transform3(imm)
                gen_pred = gender(imm_tra.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                if gen_pred[0] >= gen_pred[1]:
                    ma += 1
                else:
                    fe += 1

            case 'vit_l_16':
                imm_tra = transform2(imm)
                gen_pred = gender(imm_tra.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                if gen_pred[0] >= gen_pred[1]:
                    ma += 1
                else:
                    fe += 1

            case 'resnet152' | 'efficientnet_b7_ns':
                imm_tra = transform2(imm)
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

    paths.append(filename.split('/')[-1])

sub = pd.DataFrame({'path':paths, 'males':males, 'females':females})
sub.to_csv('data/bodies/submission_hard_gender.csv', index=False)



# calculate test detection accuracy
test_hard_path = 'data/bodies/test_hard_gender_bodies'
files_path = f'{test_hard_path}/mix/*'
files = sorted(glob(files_path))
df = pd.read_csv(f'{test_hard_path}/bodies_labels.csv')
df = df.sort_values(by='path', ascending=True)
df.to_csv(f'{test_hard_path}/bodies_labels.csv', index=False)

filename_to_class = dict(zip(df.iloc[:, 1], df.iloc[:, 0]))

all_labels, all_probs, all_preds = [], [], []
for filename in tqdm(files):
    
    img = Image.open(filename).convert('RGB')
    imm = img

    match gender_model_selection:

        # case 'EffNetb0Model' | 'EffNetV2sModel':
        #     imm_tra = transform(imm)
        #     gen_pred = gender(imm_tra.unsqueeze(0).to(device)).detach().cpu().numpy()[0][0]
        #     if gen_pred < gender_threshold:
        #         ma += 1
        #     else:
        #         fe += 1
        
        case 'vit_b_16':
            imm_tra = transform3(imm)
            gen_pred = gender(imm_tra.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
            preds = np.argmax(gen_pred)  # Predicted class labels
            labels = filename_to_class[os.path.basename(filename)]
            all_labels.append(labels)
            all_preds.append(preds)
            all_probs.append(gen_pred)

        case 'vit_l_16':
            imm_tra = transform2(imm)
            gen_pred = gender(imm_tra.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
            preds = np.argmax(gen_pred)  # Predicted class labels
            labels = filename_to_class[os.path.basename(filename)]
            all_labels.append(labels)
            all_preds.append(preds)
            all_probs.append(gen_pred)

        # case 'resnet152' | 'efficientnet_b7_ns':
        #     imm_tra = transform2(imm)
        #     tens_output = gender(imm_tra.unsqueeze(0).to(device))

        #     # Apply softmax to the tensor
        #     tens_probs = torch.softmax(tens_output, dim=1)

        #     # Convert to numpy and get predictions
        #     gen_pred = tens_probs.detach().cpu().numpy()[0]
        #     gen_pred_idx = np.argmax(gen_pred)

        #     # Update counters based on the prediction
        #     if gen_pred_idx == 0:
        #         ma += 1
        #     else:
        #         fe += 1

        # case 'deepface':
        #     gender_pred = get_deepface_gender(imm)
        #     if gender_pred == 'Man':
        #         ma += 1
        #     else:
        #         fe += 1

        # case 'insightface':
        #     faces = gender.get(np.array(imm))
        #     gen_pred = faces[0].sex  # 1 = Male, 0 = Female
        #     if gen_pred == 1:
        #         ma += 1
        #     else:
        #         fe += 1



# Convert lists to numpy arrays for log_loss and accuracy
all_labels = np.array(all_labels)  # Shape: (num_samples, )
all_probs = np.array(all_probs)  # Shape: (num_samples, 2)
# Calculate parameters
val_logloss = log_loss(all_labels, all_probs, labels=list(range(2)))
val_acc = accuracy_score(all_labels, all_preds)


preds_df = pd.DataFrame(all_preds, columns=['Predicted Label'])
probs_df = pd.DataFrame(all_probs, columns=['Prob_Male', 'Prob_Female'])

df_updated = pd.concat([df, preds_df, probs_df], axis=1)
df_updated.to_csv(f'{test_hard_path}/bodies_labels_result.csv', index=False)

print(f"Test Log Loss: {val_logloss:.4f}, Test Accuracy: {val_acc:.4f}")
