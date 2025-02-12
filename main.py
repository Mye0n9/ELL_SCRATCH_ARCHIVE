import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from models.ResNet import ResNet34
from models.FractalNet import FractalNet
from models.DenseNet import DenseNet

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

training_data = datasets.CIFAR10( # download or load CIFAR10 training dataset
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.CIFAR10( # download or load CIFAR10 training dataset
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

batch_size = 100

train_dataloader = DataLoader(training_data, batch_size=batch_size) # train DataLoader with batchsize 64
test_dataloader = DataLoader(test_data, batch_size=batch_size) # test DataLoader with batchsize 64


# model = ResNet34(num_classes = 10)
# model = FractalNet(n_blocks = 3, n_columns = 4, n_classes = 10)
model = DenseNet(n_classes=10, n_blocks=3, k=12, l=5, in_channels=16, p_dropout=0.15, compact= 0.5)

torch.cuda.set_device(1)
model.to(device)

lr = 0.02
# log_dir = './runs/ResNet'
# log_dir = './runs/FractalNet'
log_dir = './runs/DenseNet'

writer = SummaryWriter(log_dir)

def train_loop(dataloader, epoch, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    model.train()

    correct, loss = 0,0 
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()


        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")

    writer.add_scalar("Loss/train (without drop path)", loss, epoch)
    writer.add_scalar("Accuracy/train (without drop path)", correct, epoch)


def test_loop(dataloader,epoch, model, loss_fn):


    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test", correct, epoch)

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 400
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_dataloader, epoch, model, loss_fn, optimizer)
    test_loop(test_dataloader, epoch, model, loss_fn)
print("Done!")
