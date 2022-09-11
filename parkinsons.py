import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models

from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.io import read_image, ImageReadMode

from PIL import Image

import time
import os
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lib import get_files, imshow, load_model, create_data, split_train_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(device) == "cuda:0":
    print('Cuda available')
    print()


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


def train_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    val_losses = []

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

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

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

    save_losses_fig(train_losses, val_losses)

    return model


def save_losses_fig(train_losses, val_losses):
    plt.plot(np.arange(len(train_losses)), train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.savefig('results/losses.png')


def visualize_model(model, test_data, std_vec, mean_vec, num_images=6):
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(
                    f'pred/corr: {ParkinsonImageDataset.CLASSES[preds[j].item()]}, {ParkinsonImageDataset.CLASSES[labels[j].item()]}')

                implot(inputs.cpu().data[j], std_vec, mean_vec)

                if images_so_far == num_images:
                    plt.savefig('results/predictions.png')
                    return


def implot(img, std_vec, mean_vec):
    img = img * std_vec[:, None, None] + mean_vec[:, None, None]
    plt.imshow(img.numpy().transpose(1, 2, 0))


def test_model(model, test_data, test_set_sizes):
    model.eval()

    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / test_set_sizes

    print()
    print(f'Test set acc: {acc:.4f}')


class ParkinsonModel(nn.Module):

    def __init__(self):
        super().__init__()

        # model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # model_ft = models.resnet152(weights=ResNet152_Weights.DEFAULT)

        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(
            self.num_ftrs, len(ParkinsonImageDataset.CLASSES))

        self.model_ft = self.model_ft.to(device)

    def forward(self, x):
        return self.model_ft(x.to(device))


def get_transforms(std_vec, mean_vec):
    IMG_SIZE = (144, 144)

    transforms = {
        "train": torchvision.transforms.Compose([
            torchvision.transforms.Resize(IMG_SIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=mean_vec,
                std=std_vec,
            ),
        ]),
        "val": torchvision.transforms.Compose([
            torchvision.transforms.Resize(IMG_SIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=mean_vec,
                std=std_vec,
            ),
        ]),
        "test": torchvision.transforms.Compose([
            torchvision.transforms.Resize(IMG_SIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=mean_vec,
                std=std_vec,
            ),
        ])
    }

    return transforms


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_files('data/drawings')
    train_data = create_data(X_train, y_train)
    train_data, val_data = split_train_data(train_data)
    test_data = create_data(X_test, y_test)

    # plot_sample_distribution(train_data)
    # plot_sample_distribution(val_data)

    mean_vec = torch.tensor([0.485, 0.456, 0.406])
    std_vec = torch.tensor([0.229, 0.224, 0.225])

    transforms = get_transforms(std_vec, mean_vec)

    image_datasets = {
        'train': ParkinsonImageDataset(train_data, transforms['train']),
        'val': ParkinsonImageDataset(val_data, transforms['val']),
        'test': ParkinsonImageDataset(test_data, transforms['test'])
    }

    batch_size = 6

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size)
    }

    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['train', 'val', 'test']}

    model_ft = ParkinsonModel()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every x epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=5, gamma=0.1)

    model = train_model(model_ft, dataloaders, criterion, optimizer_ft,
                        exp_lr_scheduler, dataset_sizes, num_epochs=25)

    visualize_model(model, dataloaders['test'], std_vec, mean_vec, batch_size)

    test_model(model, dataloaders['test'], dataset_sizes['test'])

    torch.save(model.state_dict(), 'model.pt')
