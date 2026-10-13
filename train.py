import torch
import model
import data

train_loader,val_loader = data.create_data_loaders(data_dir='data', batch_size=32, seed=42)
network = model.create_model()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(network.parameters(),lr=0.01)
def train_one_epoch(network,train_loader,loss_fn,optimizer):
    network = model.create_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(network.parameters(),lr=0.01)

    for images, labels in train_loader:
        optimizer.zero_grad() #清空梯度
        outputs = network(images)  # 让模型计算分数
        print(outputs.shape)
        print(labels.shape)
        loss = loss_fn(outputs, labels)  #利用正确答案，评估这些分数的误差
        print(loss.item())
        loss.backward()
        optimizer.step()
        outputs = network(images) 
        loss = loss_fn(outputs, labels)
        print(loss.item())
        predictions=outputs.argmax(dim=1)
        print(predictions.shape)
        print(predictions[0].item())
        print(labels[0].item())