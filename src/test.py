import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from model import HeheNet
from utils import get_default_device, classes
from config import weight_result_path, batch_size

def evaluate(net, testloader, device):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    correct = 0
    total = 0

    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    for classname in classes:
        accuracy = 100 * correct_pred[classname] / total_pred[classname]
        print(f'=== Accuracy for class: {classname:5s} is {accuracy:.1f} % ===')

    print("=" * 50)
    print(f'🎯 Overall accuracy on 10000 test images: {100 * correct / total:.2f} %')
    print("=" * 50)

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Evaluate HeheNet on CIFAR-10 test set")
    parser.add_argument(
        "--data-path",
        type=str,
        default=weight_result_path,
        help=f"Path to trained model (.pth). Default: {weight_result_path}"
    )
    args = parser.parse_args()

    device = get_default_device()
    net = HeheNet().to(device)
    net.load_state_dict(torch.load(args.data_path, map_location=device))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size, shuffle=False, num_workers=2)

    print("=" * 50)
    print(f"Model loaded from: {args.data_path}")
    print("=" * 50)
    evaluate(net, testloader, device)


