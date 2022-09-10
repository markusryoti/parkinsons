import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models

from torchvision.models import ResNet18_Weights
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.io import read_image, ImageReadMode

from PIL import Image

import time
import os
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lib import get_files, imshow


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(device) == "cuda:0":
    print('Cuda available')


class ParkinsonImageDataset(Dataset):
    CLASSES = {0: "healthy", 1: "parkinson"}

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.labels = {"healthy": 0, "parkinson": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data.iloc[idx, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        label_value = self.data.iloc[idx, 1]

        assert (img_path.find(label_value) != -1)

        label = self.labels[label_value]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_data(X, y):
    return pd.DataFrame({'filename': X, 'healthy': y})


def split_train_data(df):
    df = df.sample(frac=1)
    num_val = int(len(df) * 0.3)

    val = df.iloc[:num_val]
    train = df.iloc[num_val:]

    assert (len(val) + len(train) == len(df))

    return train, val


def train_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
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
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def visualize_model(model, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j].item()]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    return

        model.train(mode=was_training)


def test_model(model, test_data, test_set_sizes):
    model.eval()

    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / test_set_sizes

    print()
    print(f'Test set acc: {acc:.4f}')


if __name__ == "__main__":
    transforms = {
        "train": transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
        "val": transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
        "test": transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    }

    X_train, X_test, y_train, y_test = get_files('data/drawings')
    train_data = create_data(X_train, y_train)
    train_data, val_data = split_train_data(train_data)
    test_data = create_data(X_test, y_test)

    image_datasets = {
        'train': ParkinsonImageDataset(train_data, transforms['train']),
        'val': ParkinsonImageDataset(val_data, transforms['val']),
        'test': ParkinsonImageDataset(test_data, transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=6, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=6, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=6, shuffle=True)
    }

    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['train', 'val', 'test']}
    class_names = ParkinsonImageDataset.CLASSES

    model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=1, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft,
                           exp_lr_scheduler, dataset_sizes, num_epochs=5)

    # visualize_model(model_ft, dataloaders, class_names)

    test_model(model_ft, dataloaders['test'], dataset_sizes['test'])
