#从 PNG 文件读取图片并预测
import torch
import torchvision
import model
import PIL.Image as PIL_image
import argparse

_to_tensor = torchvision.transforms.ToTensor()
def load_image(image_path):
    img=PIL_image.open(image_path).convert('L').resize((28, 28))
    return _to_tensor(img)

def predict_one(network,image):
    network.eval()
    with torch.no_grad():
        output=network(image.unsqueeze(0))
        prediction =output.argmax(dim=1).item()
    return prediction


def main():
    parser=argparse.ArgumentParser(description='Predict')
    parser.add_argument('image', type=str, nargs='?', default='examples/digit_4.png',
                    help='image path')
    args=parser.parse_args()
    image=load_image(args.image)
    
    # 加载已保存的模型
    loaded_network = model.create_model()
    loaded_network.load_state_dict(torch.load('checkpoints/mnist_model.pth', weights_only=True))  # Load the saved state dictionary into the new model instance
    
    prediction1=predict_one(loaded_network,image)
    print(f"Predicted Label: {prediction1}")

    #单张图片需要补上批次维度[1, 28, 28] → [1, 1, 28, 28]，模型输出形状为[1, 10]
    print(image.unsqueeze(0).shape)  # 输出图片的形状
   
if __name__ == "__main__":
    main()