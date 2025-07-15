import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from model import HeheNet
from utils import get_default_device
from config import num_epochs, result_path

def train(net, trainloader, device, criterion, optimizer, num_epochs=5):
    net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if i % 2000 == 1999:
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0
        epoch_loss = running_loss / len(trainloader)
        print(f"📘 Epoch [{epoch + 1:3d}/{num_epochs}] - Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Train HeheNet on CIFAR-10")
    parser.add_argument(
        "--data-path",
        type=str,
        default=result_path,
        help=f"Path to trained model (.pth). Default: {result_path}"
    )
    args = parser.parse_args()


    print(f"=== Start training HeheNet on CIFAR-10 ({num_epochs} epochs) ===")

    device = get_default_device()
    net = HeheNet().to(device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=20, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(net, trainloader, device, criterion, optimizer, num_epochs)
    torch.save(net.state_dict(), args.data_path)

    print(f"=== Save model to {args.data_path} ===")

