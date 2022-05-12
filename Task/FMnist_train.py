from BackBone.ResNet import resnet_block
from Dataset.FashionMnist import FMnist

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":

    # data loader
    train_loader = DataLoader(dataset=FMnist(train=True), batch_size=5, shuffle=True)
    test_loader = DataLoader(dataset=FMnist(train=False), batch_size=5, shuffle=True)


    

