"""Convolutional networks.

Solve any problem of your choice. The only mandatory requirement is the use of
convolutional layers (for example, image classification — if the chosen dataset has
many classes, it is enough to keep only 5, or image generation using GAN).
"""

import os
import random
import subprocess
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tqdm import tqdm

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
if torch.mps.is_available():
    DEVICE = torch.device("mps")
if torch.xpu.is_available():
    DEVICE = torch.device("xpu")


def download_dataset():
    """Download and extract the Intel Image Classification dataset from Kaggle."""
    dataset_name = "puneet6060/intel-image-classification"
    download_path = "intel-image-classification"

    if not os.path.exists(download_path):
        print("Downloading dataset from Kaggle...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_name], check=True
            )
            zip_file = "intel-image-classification.zip"
            if os.path.exists(zip_file):
                print("Extracting dataset...")
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(".")
                os.remove(zip_file)
                print("Dataset downloaded and extracted successfully!")
            else:
                print("Error: Downloaded zip file not found.")
        except subprocess.CalledProcessError:
            print("Error: Make sure you have Kaggle API installed and configured.")
            print("Run: pip install kaggle")
            print("Setup API credentials: https://www.kaggle.com/docs/api")
    else:
        print("Dataset already exists.")


def load_images(directory):
    """Load images from directory and return shuffled arrays."""
    category_map = {
        "buildings": 0,
        "forest": 1,
        "glacier": 2,
        "mountain": 3,
        "sea": 4,
        "street": 5,
    }

    images = []
    labels = []

    for category_name in os.listdir(directory):
        if category_name in category_map:
            category = category_map[category_name]
            category_path = os.path.join(directory, category_name)

            for file in os.listdir(category_path):
                img = cv2.imread(os.path.join(category_path, file))
                if img is not None:
                    resized_img = cv2.resize(img, (150, 150))
                    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                    images.append(resized_img)
                    labels.append(category)

    return shuffle(images, labels, random_state=817328462)


def get_transforms():
    """Create data augmentation transforms for train and test sets."""
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(36),
            transforms.ColorJitter(contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform


class ImageDataset(Dataset):
    """Custom Dataset for image classification."""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


class CustomCNN(nn.Module):
    """Lightweight CNN model for image classification."""

    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 150x150 -> 75x75

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 75x75 -> 37x37

            # Block 3: 64 -> 64
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 37x37 -> 18x18
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 18 * 18, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs=10, patience=10):
    """Train the model with early stopping."""
    model.to(DEVICE)
    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    history = {"train_loss": [], "train_acc": []}

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )

        # Early stopping
        if epoch_loss < best_loss - 0.001:
            best_loss = epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def evaluate_model(model, test_loader):
    """Evaluate the model on test data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def visualize_samples(images, num_samples=25):
    """Visualize random sample images."""
    rows = cols = int(np.sqrt(num_samples))
    f, ax = plt.subplots(rows, cols, figsize=(12, 12))

    for i in range(rows):
        for j in range(cols):
            rnd_number = random.randint(0, len(images) - 1)
            ax[i, j].imshow(images[rnd_number])
            ax[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the CNN training pipeline."""
    print(f"Using device: {DEVICE}")

    # Download dataset
    download_dataset()

    # Define directories
    train_dir = "seg_train/seg_train/"
    test_dir = "seg_test/seg_test/"

    # Load data
    print("Loading training data...")
    X_train, y_train = load_images(train_dir)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("Loading test data...")
    X_test, y_test = load_images(test_dir)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    # Visualize samples
    print("Visualizing samples...")
    visualize_samples(X_train)

    # Get transforms
    train_transform, test_transform = get_transforms()

    # Create datasets and dataloaders
    train_dataset = ImageDataset(X_train, y_train, transform=train_transform)
    test_dataset = ImageDataset(X_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Create model
    print("Creating model...")
    model = CustomCNN(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("Training model...")
    history = train_model(
        model, train_loader, criterion, optimizer, num_epochs=10, patience=5
    )

    # Evaluate model
    print("Evaluating model...")
    accuracy = evaluate_model(model, test_loader)

    return model, history, accuracy


if __name__ == "__main__":
    model, history, accuracy = main()
