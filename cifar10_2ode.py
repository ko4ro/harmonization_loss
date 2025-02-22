import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

from harmonization_loss import standardize_cut, pyramidal_mse_with_tokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_input_size = 64 * 16 * 16
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_center_mask(batch_size, channel, height, width):
    mask = torch.zeros((batch_size, channel, height, width)).to(device)
    grid_size = int(batch_size ** 0.5)  # 3x3 grid for batch size 9
    center_index = batch_size // 2  # Index of the center image in 3x3 grid
    mask[center_index, :, :, :] = 1  # Only the center image gets a mask
    return mask

def create_mask(batch_size, height, width, mask_size=16):
    mask = torch.zeros((batch_size, height, width)).to(device)
    start = (height - mask_size) // 2
    end = start + mask_size
    mask[:, start:end, start:end] = 1
    return mask

# 検証データの評価（Val Loss と Accuracy の記録を追加）
def evaluate_model(model, dataloader, criterion, writer, epoch):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    val_loss /= len(dataloader)
    wandb.log({"val_loss": val_loss,  "epoch": epoch})
    wandb.log({"val_acc": accuracy,  "epoch": epoch})
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', accuracy, epoch)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

def main():
  wandb.init(project="cnn-cifar10", config={
      "epochs": 20,
      "use_gradloss": True,
      "use_pyramidal_mse": False,
      "channel": 1,
      "mask_size": 16,
      "batch_size": 256,
      "learning_rate": 0.001,
      "alpha": 1.0
  })

  writer = SummaryWriter()

  num_epochs = wandb.config["epochs"]
  use_gradloss = wandb.config["use_gradloss"]
  use_pyramidal_mse = wandb.config["use_pyramidal_mse"]
  channel = wandb.config["channel"]
  mask_size = wandb.config["mask_size"]
  batch_size = wandb.config["batch_size"]
  learning_rate = wandb.config["learning_rate"]
  alpha = wandb.config["alpha"]

  if channel == 1:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Grayscale (1 channel)
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
  elif channel == 3:
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # No Grayscale conversion (preserve RGB)
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
    ])
  else:
    raise ValueError("Invalid channel setting. Only 1 (Grayscale) or 3 (RGB) is supported.")

  dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  model = CNNClassifier(num_classes=10).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

  # モデルの学習
  for epoch in range(num_epochs):
      epoch_loss = 0.0
      model.train()
      for image, label in dataloader:
          batch_size = image.size(0)
          true_heatmaps = create_mask(batch_size, 32, 32, mask_size)
          # true_heatmaps = create_center_mask(batch_size, 32, 32)
          image, label = image.to(device), label.to(device)
          image.requires_grad = True  # Enable gradient computation for images
          output = model(image)
          loss = criterion(output, label)
          if use_gradloss:
              loss_metapred = torch.sum(output * F.one_hot(label, num_classes=10), dim=-1)
              sa_maps = torch.autograd.grad(loss_metapred, image, grad_outputs=torch.ones_like(loss_metapred), retain_graph=True)[0]
              sa_maps = torch.mean(sa_maps, dim=1) # Average over channels

              # Apply the standardization-cut procedure on heatmaps
              sa_maps_preprocess = standardize_cut(sa_maps)
              heatmaps_preprocess = standardize_cut(true_heatmaps)

              if use_pyramidal_mse == True:
                # Re-normalize before pyramidal
                _sa_max = torch.amax(sa_maps_preprocess.detach(), dim=(1, 2), keepdim=True)[0] + 1e-6
                _hm_max = torch.amax(heatmaps_preprocess, dim=(1, 2), keepdim=True)[0] + 1e-6
                heatmaps_preprocess = heatmaps_preprocess / _hm_max * _sa_max
                
                # Compute and combine the pyramidal losses
                tokens = torch.ones(len(image)).to(device) # tokens are flags to indicate if the image have an associated heatmap
                # For example, if we decide to mix the ClickMe dataset with ImageNet, we may not have heatmaps for each images, in that case we can use the `tokens` flag parameters to designate when an heatmaps is provided (`1` means heatmaps provided).
                harmonization_loss = pyramidal_mse_with_tokens(sa_maps_preprocess, heatmaps_preprocess, tokens)
              else:
                # Compute and combine the losses
                harmonization_loss = F.mse_loss(sa_maps_preprocess, heatmaps_preprocess)
              
              total_loss = loss + alpha * harmonization_loss
          else:
              total_loss = loss
          optimizer.zero_grad()
          total_loss.backward()
          optimizer.step()
          epoch_loss += total_loss.item()
      average_loss = epoch_loss / len(dataloader)
      scheduler.step(average_loss)
      wandb.log({"train_loss": average_loss, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch})
      writer.add_scalar("Loss/train", average_loss, epoch)
      print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
      if (epoch + 1) % 5 == 0:
          evaluate_model(model, val_dataloader, criterion, writer, epoch)
  writer.close()
  wandb.finish()

if __name__ == "__main__":
  main()
