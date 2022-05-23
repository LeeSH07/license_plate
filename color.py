import torch
import torch.nn as nn
import os
import  numpy as np
import pandas as pd
from torch.utils.data import  Dataset, DataLoader
from  PIL import  Image
from  torchvision.transforms import  ToTensor, Compose, Resize, CenterCrop
import  glob
import color_dataset

import  numpy as np
import  torch
from PIL import  Image
from torchvision.transforms import ToTensor, Resize, CenterCrop, Compose
import  torch.nn as nn
import matplotlib.pyplot as plt
import  numpy as np

from color_dataset import VehicleColorModel

labels = ['black', 'blue' , 'cyan' , 'gray' , 'green' , 'red' , 'white' , 'yellow']
def decode_label(index):
    return  labels[index]

def encode_label_from_path(path):
    for index,value in enumerate(labels):
        if value in path:
            return  index

class VehicleColorDataset(Dataset):
    def __init__(self, image_list, class_list, transforms = None):
        self.transform = transforms
        self.image_list = image_list
        self.class_list = class_list
        self.data_len = len(self.image_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.class_list[index]

from  collections import  defaultdict
import json
class Logger(object):
    def __init__(self, log_dir, name, chkpt_interval):
        super(Logger,self).__init__()
        self.chkpt_interval = chkpt_interval
        self.log_dir = log_dir
        self.name = name
        os.makedirs(os.path.join(log_dir, name), exist_ok= True)
        self.log_path = os.path.join(log_dir, name, 'logs.json')
        self.model_path = os.path.join(log_dir, name, 'model.pt')
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0

    def log(self, key, value ):
        if isinstance(value, dict):
            for k,v in value.items():
                self.log(f'{key}.{k}',v)
        else:
            self.logs[key].append(value)

    def checkpoint(self, model):
        if (self.logs['epoch'] + 1 ) % self.chkpt_interval == 0:
            self.save(model)
        self.logs['epoch'] +=1

    def save(self, model):
        print("Saving Model...")
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, sort_keys=True, indent=4)
        epch = self.logs['epoch'] + 1
        torch.save(model.state_dict(), os.path.join(self.log_dir, self.name, f'model_{epch}.pt'))

transforms=Compose([Resize(224),  CenterCrop(224) , ToTensor()])
train_dataset = VehicleColorDataset( x_train , y_train , transforms)
train_data_loader = DataLoader(train_dataset,batch_size=115 )
test_dataset = VehicleColorDataset(x_test, y_test,transforms)
test_data_loader = DataLoader(test_dataset, batch_size=115)

from sklearn.model_selection import train_test_split

path = '/content/drive/MyDrive/color/color/'
image_list = glob.glob(path + '**/*')
class_list = [encode_label_from_path(item) for item in image_list]
x_train, x_test , y_train , y_test = train_test_split(image_list, class_list, train_size= 0.5 , stratify=class_list , shuffle=True, random_state=42)


def color_filter(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # Save Model
    # Model_Path = '/content/drive/MyDrive/Color_torch'
    # logger = Logger(Model_Path, "Exp1", 1)
    #
    # model = VehicleColorModel()
    # model.cuda()
    # opt = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, min_lr=1e-08, factor=0.1, verbose=True)
    # loss_fn = nn.CrossEntropyLoss()

    imgae_path = path
    model_path = '/content/drive/MyDrive/Color_torch/Exp1/model_5.pt'
    image = Image.open(imgae_path).convert('RGB')

    transforms = Compose([Resize(224), CenterCrop(224), ToTensor()])
    image = transforms(image)
    model = VehicleColorModel()
    model.load_state_dict(torch.load(model_path))
    t_img = image.numpy()
    # print(t_img.shape)
    plt.imshow(image.permute(1, 2, 0))
    # print(image.shape)
    image = image.unsqueeze(0)
    pred = model.forward(image).argmax(dim=1)
    class_label = decode_label(pred)
    print(class_label)

    return class_label