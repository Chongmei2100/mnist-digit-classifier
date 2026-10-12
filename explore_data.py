import torch
import torchvision
import matplotlib.pyplot as plt
import data


train_set = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=torchvision.transforms.ToTensor())
val_set = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor())
train_loader,val_loader = data.create_data_loaders(data_dir='data', batch_size=32, seed=42)
for images, labels in train_loader:
    print(images.size())
    print(labels.size())
    break
print(len(train_loader.dataset))
for images, labels in val_loader:
    print(images.size())
    print(labels.size())
    break
print(len(val_loader.dataset))
image,label = train_loader.dataset[0]
print(label)

plt.imshow(image[0], cmap='gray')
# image_tensor = torchvision.transforms.ToTensor()(image) # convert PIL image to tensor
# print(image.size())
print(image.shape)
print(image.dtype)
print(image.min())
print(image.max())
plt.show()