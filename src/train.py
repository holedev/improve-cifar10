import argparse
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from model import HeheNet
from utils import get_default_device
from config import num_epochs, weight_result_path, batch_size, chart_result_folder


def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), 100 * correct / total

def train(net, trainloader, valloader, device, criterion, optimizer, num_epochs=5):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate(net, valloader, device, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"📘 Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    return train_losses, train_accs, val_losses, val_accs

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Train HeheNet on CIFAR-10")
    parser.add_argument(
        "--data-path",
        type=str,
        default=weight_result_path,
        help=f"Path to trained model (.pth). Default: {weight_result_path}"
    )
    args = parser.parse_args()


    print(f"=== Start training HeheNet on CIFAR-10 ({num_epochs} epochs) ===")

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
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
        trainset, batch_size, shuffle=True, num_workers=2)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    valset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_losses, train_accs, val_losses, val_accs = train(
        net, trainloader, valloader, device, criterion, optimizer, num_epochs)
    
    torch.save(net.state_dict(), args.data_path)
    print(f"=== Save model to {args.data_path} ===")
    
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{chart_result_folder}/{timestamp}_loss_plot_{num_epochs}epochs.png")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{chart_result_folder}/{timestamp}_accuracy_plot_{num_epochs}epochs.png")

    print(f"=== Saved plots: {chart_result_folder}/{timestamp}_loss_plot_{num_epochs}epochs.png & {chart_result_folder}/{timestamp}_accuracy_plot_{num_epochs}epochs.png ===")

