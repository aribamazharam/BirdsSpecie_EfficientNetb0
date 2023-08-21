
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
from tqdm.notebook import tqdm

def train_birds_model(train_path, valid_path, num_epochs=10, batch_size=64, learning_rate=0.001, model_save_path=None):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_path, transform=data_transforms)
    val_dataset = ImageFolder(valid_path, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        train_loss /= len(train_dataset)
        train_accuracy = train_correct.double() / len(train_dataset)

        # Evaluation code omitted for brevity...

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)

    return model, train_losses, val_losses, train_accuracies, val_accuracies
