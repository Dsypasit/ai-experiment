from torchvision import models
import numpy as np
from torch import nn, optim
import torch
from train import train
import pickle

class TransferFeatures(nn.Module):
    def __init__(self, original_model, classifier, model_name):
        super(TransferFeatures, self).__init__()

        self.features = original_model.features
        print(self.features)
        self.classifier = classifier
        self.modelName = model_name

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        # flatten network
        f = f.view(f.size(0), np.prod(f.shape[1:]))
        y = self.classifier(f)
        return y

class MyClassification:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def save_model(self):
        torch.save(self.model, f'models/{self.__class__.__name__}.pt')
    
    def save_training_data(self, losses, accuracies, test_losses, test_accuracies):
        """
        Save training data (losses, accuracies, test_losses, and test_accuracies) to a file using pickle.

        Args:
            losses (list): List of training losses.
            accuracies (list): List of training accuracies.
            test_losses (list): List of test losses.
            test_accuracies (list): List of test accuracies.
            filename (str): Name of the file to save the data.
        """
        data = {
            'losses': losses,
            'accuracies': accuracies,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies
        }

        filename = f'data/{self.__class__.__name__}.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(data, file)

class Alexnet(MyClassification):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1028),
            nn.ReLU(inplace=True),
            nn.Linear(1028, 2),
        )

        self.refit_model = TransferFeatures(self.model, classifier, 'transfer')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.refit_model.parameters(), lr=1e-3)
    
    def train(self, train_dataloader, test_dataloader, epochs):
        self.data = train(self.refit_model, self.optimizer, self.criterion, train_dataloader, test_dataloader, epochs)
        return self.data

class VGG16(MyClassification):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 1028),
            nn.ReLU(inplace=True),
            nn.Linear(1028, 2),
        )

        self.refit_model = TransferFeatures(self.model, classifier, 'transfer')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.refit_model.parameters(), lr=1e-3)
    
    def train(self, train_dataloader, test_dataloader, epochs):
        self.data = train(self.refit_model, self.optimizer, self.criterion, train_dataloader, test_dataloader, epochs)
        return self.data

class EffNet2(MyClassification):
    def __init__(self):
        super().__init__()
        self.weights = models.EfficientNet_B1_Weights.IMAGENET1K_V2
        self.model = models.efficientnet_b1(weights=self.weights)
        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(81920, 1028),
            nn.ReLU(inplace=True),
            nn.Linear(1028, 2),
        )

        self.refit_model = TransferFeatures(self.model, classifier, 'transfer')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.refit_model.parameters(), lr=1e-3)
    
    def train(self, train_dataloader, test_dataloader, epochs):
        self.data = train(self.refit_model, self.optimizer, self.criterion, train_dataloader, test_dataloader, epochs)
        return self.data

class MobileNetV2(MyClassification):
    def __init__(self):
        super().__init__()
        self.weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.model = models.mobilenet_v3_large(weights=self.weights)
        classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(47040, 1028),
            nn.ReLU(inplace=True),
            nn.Linear(1028, 2),
        )

        self.refit_model = TransferFeatures(self.model, classifier, 'transfer')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.refit_model.parameters(), lr=1e-3)
    
    def train(self, train_dataloader, test_dataloader, epochs):
        self.data = train(self.refit_model, self.optimizer, self.criterion, train_dataloader, test_dataloader, epochs)
        return self.data