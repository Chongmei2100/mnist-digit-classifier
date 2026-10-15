import torch
import torchvision
import data
import model
from train import evaluate

def main():
    train_loader, val_loader = data.create_data_loaders(data_dir='data', batch_size=32, seed=42)
    loaded_network = model.create_model()
    loaded_network.load_state_dict(torch.load('checkpoints/mnist_model.pth', weights_only=True))  # Load the saved state dictionary into the new model instance
    loss_fn = torch.nn.CrossEntropyLoss()
    sample_num, avg_loss, acc = evaluate(loaded_network, val_loader, loss_fn)
    print(f"Sample Number: {sample_num}, Average Loss: {avg_loss}, Accuracy: {acc}")

if __name__ == "__main__":
    main()