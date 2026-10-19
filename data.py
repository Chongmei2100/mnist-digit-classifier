import torch
import torchvision
def create_data_loaders(data_dir='data', batch_size=32,seed=42):
    dataset=torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=torchvision.transforms.ToTensor())
    random_generator =torch.Generator().manual_seed(seed) # set random seed for reproducibility
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000], generator=random_generator) # split dataset into training and validation sets
    train_loader=torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    val_loader=torch.utils.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
# create data loaders for training and validation
def create_test_loader(data_dir='data', batch_size=32):
    test_set=torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_loader=torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=0)
    return test_loader