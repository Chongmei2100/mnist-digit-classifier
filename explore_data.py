import sys
import torch
import torchvision
import matplotlib.pyplot as plt

dataset=torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
print(len(dataset))
image,label = dataset[0]
print(label)
loaders = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
for images, labels in loaders:
    print(images.size())
    print(labels.size())
    break
plt.imshow(image[0], cmap='gray')
# image_tensor = torchvision.transforms.ToTensor()(image) # convert PIL image to tensor
# print(image.size())
print(image.shape)
print(image.dtype)
print(image.min())
print(image.max())
plt.show()