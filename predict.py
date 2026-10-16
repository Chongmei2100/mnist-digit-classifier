#加载已保存的模型，从验证集中取一张图片进行预测，并输出预测数字和真实标签
import torch
import torchvision
import data
import model
from train import evaluate

def main():
    # 创建数据加载器
    train_loader, val_loader = data.create_data_loaders(data_dir='data', batch_size=32, seed=42)
    
    # 加载已保存的模型
    loaded_network = model.create_model()
    loaded_network.load_state_dict(torch.load('checkpoints/mnist_model.pth', weights_only=True))  # Load the saved state dictionary into the new model instance
    
    # 从验证集中取一张图片进行预测
    image, label = val_loader.dataset[0]
    loaded_network.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = loaded_network(image.unsqueeze(0))  # Add batch dimension
        prediction = output.argmax(dim=1).item()  # Get the predicted class index  

    #单张图片需要补上批次维度[1, 28, 28] → [1, 1, 28, 28]，模型输出形状为[1, 10]
    print(image.unsqueeze(0).shape)  # 输出图片的形状
    print(output.shape)  # 输出模型输出的形状
    
    print(f"Predicted Label: {prediction}, True Label: {label}")
if __name__ == "__main__":
    main()