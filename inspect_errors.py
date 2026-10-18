import torch
import torchvision
import matplotlib.pyplot as plt
import model
import data
from predict import predict_one

def main():
    #加载验证集
    train_loader,val_loader = data.create_data_loaders(data_dir='data', batch_size=32, seed=42)
    #创建网络并加载保存的参数
    loaded_network = model.create_model()
    loaded_network.load_state_dict(torch.load('checkpoints/mnist_model.pth', weights_only=True))  # Load the saved state dictionary into the new model instance
    #
    cnt=0
    for i in range(len(val_loader.dataset)):
        image,label=val_loader.dataset[i]
        prediction=predict_one(loaded_network, image)
        if  prediction!= label:
            cnt+=1
            plt.imshow(image[0], cmap='gray')
            plt.title(f"true={label}, pred={prediction}")
            print(f"索引{i}，真实标签：{label}，预测数字：{prediction}")
            plt.show()
            
            if cnt>=3:
                break

if __name__ == '__main__':
    main()