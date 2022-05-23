import numpy as np
import scipy.io as sio
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import random
from console_progressbar import ProgressBar

from __future__ import print_function, division
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# final_model = torch.load('./data/final_make_70', map_location='cpu')

import tarfile


def extract(tar_file, path):
    opened_tar = tarfile.open(tar_file)

    if tarfile.is_tarfile(tar_file):
        opened_tar.extractall(path)
    else:
        print("The tar file you entered is not a tar file")


def get_make(class_name):
    idx = class_name.find(' ')
    return class_name[:idx]


meta = sio.loadmat('C:\Users\hhh\PycharmProjects\pythonProject2\devkit\cars_annos.mat')
make_list = []
convert = {}

number_to_make = {}
for i, clas in enumerate(meta['class_names'][0]):

    make = get_make(clas[0])
    if make not in make_list:
        make_list.append(make)
    convert[i] = make_list.index(make)
    number_to_make[convert[i]] = make

def get_labels(path):
    annos = sio.loadmat('C:\Users\hhh\PycharmProjects\pythonProject2\devkit/cars_train_annos.mat')
    _, total_size = annos["annotations"].shape
    labels = {}
    for i in range(total_size):
        path = annos['annotations'][:,i][0][5][0]
        clas = annos['annotations'][:,i][0][4][0][0]
        labels[path] = convert[clas-1]
    return labels
labels_n= get_labels('cars_train_annos')


def save_data(fnames, labels, bboxes, data_type='train'):
    if data_type == 'train':
        src_folder = '/content/drive/MyDrive/input/cars_train/cars_train'
    else:
        src_folder = '/content/drive/MyDrive/input/cars_test/cars_test'
    num_samples = len(fnames)

    if data_type == 'train':
        perm = np.random.permutation(num_samples)

        train_split = 0.8
        num_train = int(round(num_samples * train_split))

    pb = ProgressBar(total=100, prefix='Save train data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        if data_type == 'train':
            label = labels[i]
        (x1, y1, x2, y2) = bboxes[i]

        src_path = os.path.join(src_folder, fname)
        src_image = cv2.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print("{} -> {}".format(fname, label))
        pb.print_progress_bar((i + 1) * 100 / num_samples)
        dst_path = os.path.join('/content/drive/MyDrive/data/test', fname)
        if data_type == 'train':
            if i < num_train:
                dst_folder = '/content/drive/MyDrive/data/train/'
            else:
                dst_folder = '/content/drive/MyDrive/data/valid/'

            dst_path = os.path.join(dst_folder, label)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            dst_path = os.path.join(dst_path, fname)

        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv2.resize(src=crop_image, dsize=(224, 224))
        cv2.imwrite(dst_path, dst_img)


def process_data(data_type='train'):
    print("Processing train data...")
    if data_type == 'train':
        cars_annos = sio.loadmat('/content/drive/MyDrive/input/devkit/cars_train_annos.mat')
    else:
        cars_annos = sio.loadmat('/content/drive/MyDrive/input/devkit/cars_test_annos.mat')
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    class_ids = []
    bboxes = []
    labels = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]

        if data_type == 'train':
            class_id = annotation[0][4][0][0]
            fname = annotation[0][5][0]
            labels.append('%02d' % labels_n[fname])
            class_ids.append(labels_n[fname])

        else:
            fname = annotation[0][4][0]

        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        fnames.append(fname)

    if data_type == 'train':
        labels_count = np.unique(class_ids).shape[0]
        print(np.unique(class_ids))
        print('The number of different cars is %d' % labels_count)
    else:
        print('Test data processing')
    save_data(fnames, labels, bboxes, data_type)

process_data('train')
process_data('test')

bs = 64

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/content/drive/MyDrive/data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # clip
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                print(preds[j])
                ax.set_title('predicted: {}'.format(make_list[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()

model_conv = torchvision.models.resnext101_32x8d(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
# fine tuning last layer of resnext pretrained model
model_conv.fc = nn.Sequential(
                                nn.Linear(num_ftrs, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
                                nn.Linear(512, 49)
                                )
model_conv = model_conv.to(device)
criterion = FocalLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.Adam(model_conv.fc.parameters())

exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer_conv, gamma=0.96)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=100)

torch.save(model_conv, '/content/drive/MyDrive/save_file_apperance')
model = torch.load('/content/drive/MyDrive/save_file_apperance')

for name, param in model.named_parameters():
    if name in ['fc.3.weight','fc.3.bias']:
        param.requires_grad = False

model = model.to(device)
criterion = FocalLoss()
optimizer_conv = optim.Adam(model.fc.parameters(), lr=0.00003, weight_decay = 1e-9)
exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer_conv, gamma=0.96)

model = train_model(model, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=50)

torch.save(model, '/content/drive/MyDrive/appearance')
model = torch.load('/content/drive/MyDrive/appearance')

from torch.autograd import Variable
def image_loader(img):
    image = data_transforms['valid'](img).float()
    plt.imshow(img)
    image = torch.Tensor(image)
    image = image.unsqueeze(0)
    return image.cuda()  #assumes that you're using GPU


from PIL import Image, ImageFile

num_samples, all_preds = 8041, []
out = open('result.txt', 'a')
for i in range(num_samples):
    filename = os.path.join('/content/drive/MyDrive/input/cars_test/cars_test', '%05d.jpg' % (i + 1))
    bgr_img = cv2.imread(filename)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(rgb_img)

    im_pill = image_loader(im_pil)
    preds = model(im_pill)

    class_id = preds.max(1)[1]
    all_preds.append(class_id)
    out.write('{}\n'.format(str(class_id + 1)))

out.close()

labels = sio.loadmat('/content/drive/MyDrive/input/devkit/cars_test_annos_withlabels.mat')

a = []
for lab in labels['annotations']['class'][0]:
#     print(lab)
    a.append(convert[lab[0][0]-1])
actual_preds = np.array(a,dtype=np.int);
actual_preds = actual_preds.squeeze()
all_preds = np.array(all_preds)

print('accuracy = ',(all_preds==actual_preds).sum()/len(actual_preds))

from sklearn.metrics import f1_score
print("f1 score {}".format(f1_score(actual_preds.tolist(), all_preds.tolist(), average='macro')))
print(classification_report(actual_preds.tolist(), all_preds.tolist(), target_names=make_list, labels=make_list, zero_division=1))

final_model = torch.load('/content/drive/MyDrive/appearance', map_location='cpu')
import ssl
import base64
from PIL import Image, ImageFile
import http.client as httplib

headers = {"Content-type": "application/json",
           "X-Access-Token": "yrkuYbYWugkjcM3tfpO4ffCGHHOYgaJehWOD"}


def plotting(path, x1, y1, x2, y2):
    src_image = cv2.imread(path)
    crop_image = src_image[y1:y2, x1:x2]
    plt.imshow(crop_image[:, :, ::-1])


def get_box(path):
    image_data = base64.b64encode(open(path, 'rb').read()).decode()
    params = json.dumps({"image": image_data})

    conn = httplib.HTTPSConnection("dev.sighthoundapi.com",
                                   context=ssl.SSLContext(ssl.PROTOCOL_TLSv1_2))

    conn.request("POST", "/v1/recognition?objectType=vehicle", params, headers)
    response = conn.getresponse()
    result = response.read()
    json_obj = json.loads(result)

    if 'reasonCode' in json_obj and json_obj['reasonCode'] == 50202:
        print(json_obj)
        return 'TL'
    if not json_obj or 'objects' not in json_obj or len(json_obj['objects']) < 1:
        return False

    annot = json_obj['objects'][0]['vehicleAnnotation']
    vertices = annot['bounding']['vertices']
    xy1 = vertices[0]
    xy3 = vertices[2]
    return xy1['x'], xy1['y'], xy3['x'], xy3['y']


def crop_car(src_path, x1, y1, x2, y2):
    src_image = cv2.imread(src_path)
    if src_image is None:
        return
    crop_image = src_image[y1:y2, x1:x2]
    dst_img = cv2.resize(src=crop_image, dsize=(224, 224))
    img = Image.fromarray(dst_img)
    image = data_transforms['valid'](img).float()
    image = torch.Tensor(image)
    return image.unsqueeze(0)

def predict_make(src):
    resp = get_box(src)
    if not resp:
        return "error"
    plotting(src, *resp)
    image = crop_car(src, *resp)
    preds = final_model(image)
    return make_list[int(preds.max(1)[1][0])]

import json
predict_make('/content/drive/MyDrive/data2.png')
