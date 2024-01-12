import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from data_loader import load_data, split_dataset
from model import SpectrumClassifier

# Hyperparameters
NUM_EPOCHS = 100
LEARNING_RATE = 0.03
BATCH_SIZE = 32
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15

def train_model(model, criterion, optimizer, dataloaders, num_epochs=NUM_EPOCHS):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    import matplotlib.pyplot as plt

    plt.figure(figsize=[15,5])
    plt.subplot(1,2,1)
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validataion accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('Training_Process.png')

    return model

# Load Data
data, labels = load_data()


# Create Dataloaders
dataloaders = create_dataloader(data, labels, batch_size=BATCH_SIZE,
                                train_ratio=TRAIN_RATIO, 
                                valid_ratio=VALID_RATIO)

# Create a Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create Model
model = SERSModel()
model = model.to(device)

# Create Criterion
criterion = nn.CrossEntropyLoss()

# Create Optimizer
optimizer = torch.optim.Adam(model.parameters())

# Train Model
model = train_model(model, dataloaders)

# Save Model
torch.save(model.state_dict(), 'spectrum_classifier.pth')