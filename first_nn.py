import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Transformations: normalization and converting to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loades training and test data
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# Defines the CNN model with corrected input size for fc1
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Calculate the output size after convolution layers
        self._to_linear = None
        self.convs = nn.Sequential(self.conv1, self.conv2)
        self._get_conv_output()

        self.fc1 = nn.Linear(self._to_linear, 128)  # Update input size
        self.fc2 = nn.Linear(128, 10)

    def _get_conv_output(self):
        # Pass a dummy tensor through conv layers to get output size
        with torch.no_grad():
            dummy_x = torch.zeros(1, 3, 32, 32)
            dummy_x = self.convs(dummy_x)
            self._to_linear = dummy_x.shape[1] * dummy_x.shape[2] * dummy_x.shape[3]

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Instantiates the model, defines loss function and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Define number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

model.eval()  # Sets model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Using Scikit-Learn to print accuracy and classification report
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Gets some random test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Displays images and predictions
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Displays images with predictions
imshow(torchvision.utils.make_grid(images[:4]))
print("Predicted:", ' '.join(f'{test_dataset.classes[predicted[j]]}' for j in range(4)))
print("Actual:   ", ' '.join(f'{test_dataset.classes[labels[j]]}' for j in range(4)))
