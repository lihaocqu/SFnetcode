import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from model import RNN
from data_preprocessing import DatasetLoader

def train(model, train_data, epochs, criterion, optimizer):
    losses = []
    for epoch in range(epochs):
        model.zero_grad()
        output = model(torch.from_numpy(train_data).float())
        loss = criterion(output, torch.from_numpy(train_data).float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
    return losses

def test(model, test_data, criterion):
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(test_data).float())
        loss = criterion(output, torch.from_numpy(test_data).float())
        print(f'Test Loss: {loss.item()}')
    model.train()

def main():
    epochs = 10000
    learning_rate = 0.001
    dataset_filepath = 'dataset.csv'
    
    loader = DatasetLoader(dataset_filepath)
    loader.load_data()
    train_data, test_data = loader.get_data()
    
    model = RNN()
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    losses = train(model, train_data, epochs, criterion, optimizer)
    test(model, test_data, criterion)
    
if __name__ == "__main__":
    main()