import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_train_pics(path, dataset_name, batch_size=64, num_workers=2):
    """
    dataset_name 可选: mnist, cifar10, cifar100
    batch_size: 批量大小
    num_workers: 数据加载线程数
    """
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:  # cifar10和cifar100是RGB图像
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    mnist_train_dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
    mnist_test_dataset = datasets.MNIST(path, train=False, download=True, transform=transform)

    cifar10_train_dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform)
    cifar10_test_dataset = datasets.CIFAR10(path, train=False, download=True, transform=transform)

    cifar100_train_dataset = datasets.CIFAR100(path, train=True, download=True, transform=transform)
    cifar100_test_dataset = datasets.CIFAR100(path, train=False, download=True, transform=transform)
    
    if dataset_name == 'mnist':
        train_loader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
    elif dataset_name == 'cifar10':
        train_loader = DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(cifar10_test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
    elif dataset_name == 'cifar100':
        train_loader = DataLoader(cifar100_train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(cifar100_test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
    else:
        raise ValueError("Unsupported dataset_name. Choose from: mnist, cifar10, cifar100.")

    return train_loader, test_loader


