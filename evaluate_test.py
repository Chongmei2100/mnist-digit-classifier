import torch
import data
import model
from train import evaluate

def main():
    test_loader = data.create_test_loader(data_dir='data', batch_size=32)
    loaded_network = model.create_model()
    loaded_network.load_state_dict(torch.load('checkpoints/mnist_model.pth', weights_only=True))  # Load the saved state dictionary into the new model instance
    loss_fn = torch.nn.CrossEntropyLoss()
    sample_num, avg_loss, acc = evaluate(loaded_network, test_loader, loss_fn)
    print(f"test_results: \n Sample Number: {sample_num}, Average Loss: {avg_loss}, Accuracy: {acc}")

if __name__ == "__main__":
    main()