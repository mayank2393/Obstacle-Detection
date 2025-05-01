import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn_model import cnn_model
from rcnn_model import rcnn_model
from training_dataset import training_dataset
from functions import overlapScore

def train_model(net, dataloader, batchSize, lr_rate, momentum, model_name):
    criterion = nn.MSELoss()
    optimization = optim.SGD(net.parameters(), lr=lr_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimization, step_size=30, gamma=0.1)

    for epoch in range(50):
        scheduler.step()
        for i, data in enumerate(dataloader):
            optimization.zero_grad()

            inputs, labels = data
            inputs, labels = inputs.view(batchSize, 1, 100, 100), labels.view(batchSize, 4)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimization.step()

            pbox = outputs.detach().numpy()
            gbox = labels.detach().numpy()
            score, _ = overlapScore(pbox, gbox)

            print(f'[epoch {epoch+1}, step: {i+1}, loss: {loss.item():.6f}, Average Score = {score/batchSize:.6f}]')

    print(f'Finish Training {model_name}')

if __name__ == '__main__':
    # Hyperparameters
    learning_rate = 0.000001
    momentum = 0.9
    batch = 100
    no_of_workers = 2
    shuffle = True

    trainingdataset = training_dataset()
    dataLoader = DataLoader(
        dataset=trainingdataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=no_of_workers
    )

    # Train CNN model
    cnn_model_instance = cnn_model()
    cnn_model_instance.train()
    train_model(cnn_model_instance, dataLoader, batch, learning_rate, momentum, "CNN")

    # Train RCNN model
    rcnn_model_instance = rcnn_model()
    rcnn_model_instance.train()
    train_model(rcnn_model_instance, dataLoader, batch, learning_rate, momentum, "RCNN")
    
    torch.save(cnn_model_instance.state_dict(), './Model/cnn_model.pth')
    torch.save(rcnn_model_instance.state_dict(), './Model/rcnn_model.pth')
