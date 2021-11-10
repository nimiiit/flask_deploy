import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

class Network(nn.Module):
  def __init__(self, classes=2):
    super(Network,self).__init__()
    self.classes=classes
    self.FeatureExtractor = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 3D convolution layer
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 4D convolution layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
    self.Pool=nn.AdaptiveAvgPool2d(1)
    self.Classifier = nn.Sequential(
            nn.Linear(64,16),
            nn.Dropout(0.2))    
    if self.classes == 2:
            self.Classifierhead = nn.Linear(16, 1)
    else:
            self.Classifierhead= nn.Linear(16, self.classes)
 

  def forward(self,x):
      x=self.FeatureExtractor(x)
      x=self.Pool(x)
      x = x.view(-1, 64)
      x=self.Classifier(x)
      x=self.Classifierhead(x)
      return x

device=torch.device('cpu')
model=Network(2)
PATH='app/Best_GenderModel.pth'
model.load_state_dict(torch.load(PATH, map_location=device)) 
model.eval()

def transform_image(image_bytes):
    img_size=64
    transform=transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])
    image= Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    predictions=model(image_tensor)
    pred_idx = torch.round(torch.sigmoid(predictions)).squeeze(1)
    return pred_idx