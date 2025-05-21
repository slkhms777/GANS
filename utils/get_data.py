import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_train_pics(path, dataset_name):
    """
    dataset_name 可选: mnist, cifar10, cifar100
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_train_dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
    mnist_test_dataset = datasets.MNIST(path, train=False, download=True, transform=transform)

    cifar10_train_dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform)
    cifar10_test_dataset = datasets.CIFAR10(path, train=False, download=True, transform=transform)

    cifar100_train_dataset = datasets.CIFAR100(path, train=True, download=True, transform=transform)
    cifar100_test_dataset = datasets.CIFAR100(path, train=False, download=True, transform=transform)
    
    if dataset_name == 'mnist':
        train_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(mnist_test_dataset, batch_size=64, shuffle=False, num_workers=2)
    elif dataset_name == 'cifar10':
        train_loader = DataLoader(cifar10_train_dataset, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(cifar10_test_dataset, batch_size=64, shuffle=False, num_workers=2)
    elif dataset_name == 'cifar100':
        train_loader = DataLoader(cifar100_train_dataset, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(cifar100_test_dataset, batch_size=64, shuffle=False, num_workers=2)
    else:
        raise ValueError("Unsupported dataset_name. Choose from: mnist, cifar10, cifar100.")

    return train_loader, test_loader


