import torch
import os
from data_management.custom_dataset import DatasetCINEDE
from models.model import FIUNet
from models.train_model import train_model
from data_management.utils import DiceLoss

PATH = "C:/Users/Cylia/Desktop/datasets/CINEDE"
N_EPOCHS = 5
device = torch.device("cpu")

dataset = DatasetCINEDE(folder_path=PATH)
torch.manual_seed(1)
train_set, val_set = torch.utils.data.dataset.random_split(dataset, [500, 99])
trainLoader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
valLoader = torch.utils.data.DataLoader(val_set, batch_size=99, shuffle=False)

# build model and train
model = FIUNet(3).to(device)
criterion = DiceLoss(use_background=True).to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainCost, valCost = train_model(n_epochs=N_EPOCHS, model=model, trainloader=trainLoader, valloader=valLoader,
                                 criterion=criterion, optimizer=optimizer)

print('this is the evolution of trainCost in {} epochs :\n'.format(N_EPOCHS))
print(trainCost)
print('\nthis is the evolution of valCost in {} epochs :\n'.format(N_EPOCHS))
print(valCost)
