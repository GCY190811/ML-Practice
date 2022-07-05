import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from BackBone.ResNet import ResNet
from BackBone.MobileNet import MobileNet
from BackBone.MobileNetV3 import MobileNetV3
from Dataset.FashionMnist import FMnist

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

if __name__ == "__main__":

    # data loader
    train_loader = DataLoader(dataset=FMnist(train=True), batch_size=5, shuffle=True)
    test_loader = DataLoader(dataset=FMnist(train=False), batch_size=5, shuffle=True)
    transforms = T.Resize([224, 224])

    # loss and optimizer
    device = torch.device('cuda:0')
    # model = ResNet().to(device)
    # model = MobileNet().to(device)
    model = MobileNetV3(num_class=10).to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training
    epochs = 5
    for epoch in range(epochs):
        sum_loss = 0.0
        train_correct = 0
        for img, label in train_loader:
            img = torch.tensor(img, dtype=torch.float32, device="cuda:0")
            label = torch.tensor(label, dtype=torch.uint8, device="cuda:0")
            img = transforms(img)

            optimizer.zero_grad()
            output = model(img)
            loss = cost(output, label)
            loss.backward()
            optimizer.step()

            _, id = torch.max(output.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == label.data)

        print('[%d/%d] loss:%.03f' % (epoch+1, epochs, sum_loss/len(train_loader)))
        print(' correct:%.03f%%' % (100 * train_correct/len(FMnist(train=True))))

        test_loss = 0
        test_correct = 0
        for img, label in test_loader:
            img = torch.tensor(img, dtype=torch.float32, device="cuda:0")
            label = torch.tensor(label, dtype=torch.uint8, device="cuda:0")
            img = transforms(img)

            model.eval()
            output = model(img)
            loss = cost(output, label)

            _, id = torch.max(output.data, 1)
            test_loss += loss.data
            test_correct += torch.sum(id == label.data)

        print('[%d/%d] loss:%.03f' % (epoch+1, epochs, test_loss/len(test_loader)))
        print(' correct:%.03f%%' % (100 * test_correct/len(FMnist(train=False))))









