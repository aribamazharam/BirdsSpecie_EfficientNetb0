
from birds_functions import train_birds_model
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def main():
    # Example usage:

    # 1. Train the model
    train_path = "/path/to/train_data"
    valid_path = "/path/to/valid_data"
    model_save_path = "/path/to/save/model.pth"
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_birds_model(train_path, valid_path, model_save_path=model_save_path)

    # Plotting the training and validation curves
    num_epochs = len(train_accuracies)
    epochs = range(1, num_epochs+1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
