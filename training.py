import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and configuration
CONFIG = {
    'DATA_DIR': 'data',
    'MODEL_DIR': 'models',
    'BATCH_SIZE': 32,
    'EPOCHS': 10,
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 0.0005,
    'LOG_INTERVAL': 100,
}

class AgentDataset(Dataset):
    def __init__(self, data_dir: str, transform: transforms.Compose):
        self.data_dir = data_dir
        self.transform = transform
        self.data = pd.read_csv(os.path.join(data_dir, 'data.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data.iloc[idx]
        image = self.transform(sample['image'])
        label = torch.tensor(sample['label'])
        return image, label

class AgentModel(nn.Module):
    def __init__(self):
        super(AgentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AgentTrainer:
    def __init__(self, model: AgentModel, device: torch.device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])

    def train(self, train_loader: DataLoader, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            logging.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')
            if (epoch + 1) % CONFIG['LOG_INTERVAL'] == 0:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch: int):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, os.path.join(CONFIG['MODEL_DIR'], f'agent_model_{epoch}.pth'))

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Load data
    data_dir = CONFIG['DATA_DIR']
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = AgentDataset(data_dir, transform)
    train_loader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)

    # Create model and trainer
    model = AgentModel()
    model.to(device)
    trainer = AgentTrainer(model, device)

    # Train model
    trainer.train(train_loader, CONFIG['EPOCHS'])

if __name__ == '__main__':
    main()