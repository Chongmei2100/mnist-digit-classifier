import torch
import model
import data

def train_one_epoch(network,train_loader,loss_fn,optimizer):
    network.train()
    processed_num = 0
    total_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad() #清空梯度
        outputs = network(images)  # 让模型计算分数
        loss = loss_fn(outputs, labels)  #利用正确答案，评估这些分数的误差
        loss.backward()
        optimizer.step()
        num=images.shape[0]
        total_loss+=num*loss.item()
        processed_num+=num
    average_loss=total_loss/processed_num
    return processed_num,average_loss

def evaluate(network,val_loader,loss_fn):
    network.eval()
    total_loss=0
    sample_num=0
    correct_num=0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = network(images)  # 让模型计算分数
            loss = loss_fn(outputs, labels)  #利用正确答案，评估这些分数的误差
            predictions=outputs.argmax(dim=1)
            num=images.shape[0]
            sample_num+=num
            correct_num += (predictions == labels).sum().item()
            total_loss+=num*loss.item()

    return sample_num, total_loss/sample_num, correct_num/sample_num

def main():
    train_loader,val_loader = data.create_data_loaders(data_dir='data', batch_size=32, seed=42)
    network = model.create_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(network.parameters(),lr=0.01)

    for epoch in range(3):
        print(f"Epoch {epoch+1}")
        a, b = train_one_epoch(network, train_loader, loss_fn, optimizer)
        print(f"Training - Sample Number: {a}, Average Loss: {b}")
        sample_num, avg_loss, acc = evaluate(network,val_loader,loss_fn)
        print(f"Sample Number: {sample_num}")
        print(f"Average Loss: {avg_loss}")
        print(f"Accuracy: {acc*100:.2f}%")
    #network.load_state_dict(torch.load('model.pth'))  # Load the model state dictionary from the file
    torch.save(network.state_dict(), 'checkpoints/mnist_model.pth')  # Save the model state dictionary to a file
    loaded_network = model.create_model()  # Create a new instance of the model
    loaded_network.load_state_dict(torch.load('checkpoints/mnist_model.pth', weights_only=True))  # Load the saved state dictionary into the new model instance
    print(evaluate(loaded_network, val_loader, loss_fn))  # Evaluate the loaded model on the validation set

if __name__ == "__main__":
    main()

