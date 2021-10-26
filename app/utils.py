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
    #loads pretrained vgg model
    vgg=models.vgg16(pretrained=True)
    self.FeatureExtractor=vgg.features
    self.numfeat=512              
    self.Pool=nn.AdaptiveAvgPool2d(1)
    #changs the classifier based on number of classes
    self.Classifier = nn.Sequential(
            nn.Linear(self.numfeat,128),
            nn.Dropout(0.2))    
    if self.classes == 2:
            self.Classifierhead = nn.Linear(128, 1)
    else:
            self.Classifierhead= nn.Linear(128, self.classes)
    #additional auxillary classifier for debiasing
    self.Aux_Classifier=nn.Sequential(
           nn.Linear(self.numfeat,128),
           nn.Dropout(0.2), nn.Linear(128,2)) 
 

  def forward(self,x):
      x=self.FeatureExtractor(x)
      x=self.Pool(x)
      x = x.view(-1, self.numfeat)
      x1=self.Classifier(x)
      x1=self.Classifierhead(x1)
      x2=self.Aux_Classifier(x)
      return x1,x2

device=torch.device('cpu')
model=Network(2)
PATH='app/Best_GenderModel.pth'
model.load_state_dict(torch.load(PATH, map_location=device)) 
model.eval()

def transform_image(image_bytes):
    img_size=128
    transform=transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])
    image= Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    predictions,aux=model(image_tensor)
    pred_idx = torch.round(torch.sigmoid(predictions)).squeeze(1)
    return pred_idx