import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
#import pretrainedmodels
#import pretrainedmodels.utils as utils
import torchvision.models as models

# Hyperparameters
num_epochs = 50
batch_size = 128
learning_rate = 0.001

class IMAGENET():
    def __init__(self, arch):
        self.model = models.__dict__[arch](pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=[0])
 
    def predict(self, image):
        image = torch.clamp(image, -1, 1)
        image = Variable(image, volatile=True).view(1,3,224,224)
        output = self.model(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_batch(self, image):
        image = torch.clamp(image, -1 ,1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self.model(image)
        _, predict = torch.max(output.data, 1)
        return predict



class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(3200,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers=[]
        in_channels= 3
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.Conv2d(128, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)


    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True).view(1,3, 32,32)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict



class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(1024,200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200,200)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(200,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers=[]
        in_channels= 1
        layers += [nn.Conv2d(in_channels, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.Conv2d(32, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)


    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True).view(1,1,28,28)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]

    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict



class SimpleMNIST(nn.Module):
    """ Custom CNN for MNIST
        stride = 1, padding = 2
        Layer 1: Conv2d 5x5x16, BatchNorm(16), ReLU, Max Pooling 2x2
        Layer 2: Conv2d 5x5x32, BatchNorm(32), ReLU, Max Pooling 2x2
        FC 10
    """
    def __init__(self):
        super(SimpleMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def predict(self, image):
        self.eval()
        image = Variable(image.unsqueeze(0))
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]


def show_image(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def load_mnist_data():
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data/mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

def load_cifar10_data():
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    # CIFAR10 Dataset
    train_dataset = dsets.CIFAR10('./data/cifar10-py', download=True, train=True, transform= transforms.ToTensor())
    test_dataset = dsets.CIFAR10('./data/cifar10-py', download=True, train=False, transform= transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

def load_imagenet_data():
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                         
    # train_dataset = dsets.ImageFolder(
    #     '/data/train',
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    val_dataset = dsets.ImageFolder(
        '/data/val',
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # Data Loader (Input Pipeline)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=True)

    return val_loader, val_loader, val_dataset, val_dataset


class ImagenetTestDataset(Dataset):
    def __init__(self, root_file, transform=None):
       self.label =[]
       self.root_dir = root_file
       self.transform = transform
       self.img_name = sorted(os.listdir(root_file))
       for img in self.img_name:
           name = img.split('.')
           self.label.append(int(name[0])-1)

    def __getitem__(self, idx):
        image = Image.open(self.root_dir + '/' + self.img_name[idx])
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        #label = torch.LongTensor(self.label[idx])
        label = self.label[idx] 
        return image, label

    def __len__(self):
        return len(self.label)

def imagenettest():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    #test_dataset = ImagenetTestDataset('/data/test')

    test_dataset = ImagenetTestDataset('/data/test', transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,]))

    # Data Loader (Input Pipeline)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

    return test_loader, test_dataset



def train_simple_mnist(model, train_loader):
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
        
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, loss.data[0]))


def train_mnist(model, train_loader):
    # Loss and Optimizer
    model.train()
    lr = 0.01
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            images = Variable(images)
            labels = Variable(labels)
        
            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, loss.data[0]))


def test_minst(model, test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct / total))


def train_cifar10(model, train_loader):
    # Loss and Optimizer
    model.train()
    lr = 0.01
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # Train the Model
    for epoch in range(num_epochs):
        if epoch%10==0 and epoch!=0:
            lr = lr * 0.95
            momentum = momentum * 0.5
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            images = Variable(images)
            labels = Variable(labels)
        
            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, loss.data[0]))

def test_cifar10(model, test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100.0 * correct / total))

class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr
    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor

class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255
    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

def save_model(model, filename):
    """ Save the trained model """
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    """ Load the training model """
    model.load_state_dict(torch.load(filename))

if __name__ == '__main__':
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    net = MNIST()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    train_mnist(net, train_loader)
    #load_model(net, 'models/mnist_gpu.pt')
    #load_model(net, 'models/mnist.pt')
    test(net, test_loader)
    save_model(net,'./models/mnist.pt')
    #net.eval()

