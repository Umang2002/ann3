import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader





def mlp_apply(model, test_loader, test_indexes):
    # Get the device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Extract the 10 test examples
    test_images = []
    test_labels = []
    for i, (images, labels) in enumerate(test_loader):
        if i in test_indexes:
            test_images.extend(images)
            test_labels.extend(labels)
        if len(test_images) >= 10:
            break
    
    test_images = torch.stack(test_images[:10]).to(device)
    test_labels = torch.stack(test_labels[:10]).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(test_images.view(test_images.size(0), -1))
        _, predicted = torch.max(outputs, 1)
    
    # Plot the images with true and predicted labels
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(test_images[i].cpu().numpy().reshape(28, 28), cmap='gray')
        plt.title(f'True: {test_labels[i].item()}\nPred: {predicted[i].item()}')
        plt.axis('off')
    plt.show()

def mlp_train(model, train_loader, test_loader, criterion, optimizer, num_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(test_loader))
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Accuracy: {accuracy:.2f}%')

    return model, train_losses, test_losses
