import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn.functional as F

# Hyperparameters
num_epochs = 50
batch_size = 128
learning_rate = 0.001

def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def load_data():
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    # MNIST Dataset
    #train_dataset = dsets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    #test_dataset = dsets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    #transform_train = tfs.Compose([
    #    tfs.RandomCrop(32, padding=4),
    #    tfs.RandomHorizontalFlip(),
    #    tfs.ToTensor()
    #    ])
 
    train_dataset = dsets.CIFAR10('./cifar10-py', download=False, train=True, transform= transforms.ToTensor())
    test_dataset = dsets.CIFAR10('./cifar10-py', download=False, train=False, transform= transforms.ToTensor())

    
    
    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_dataset, test_dataset


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        image = Variable(image).cuda().view(1,3, 32,32)
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]

   

def train(model, train_loader):
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

def test(model, test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100.0 * correct / total))

def save_model(model, filename):
    """ Save the trained model """
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    """ Load the training model """
    model.load_state_dict(torch.load(filename))

def attack(model, dataset, x0, y0, alpha = 0.018, beta = 0.05, iterations = 1000):

    if (model.module.predict(x0) != y0):
        print("Fail to classify the image. No need to attack.")
        return

    timestart = time.time()
    print("Searching for the initial direction: ")
    mindist = float('inf')
    theta = None
    info = None
    for i, (xi, yi) in enumerate(dataset):
        diff = xi-x0
        dist = torch.norm(xi-x0)
        if random.random() < 0.1 and model.module.predict(xi) != y0 and dist < mindist:
            mindist = dist
            theta, info = diff, (i, yi)

    timeend = time.time()
    print("-----> Found image indexing at %d labelled %d within distance %.4f from original image in %.4f seconds" % (info[0], info[1], mindist, timeend-timestart))

    timestart = time.time()

    query_count = 0
    g1 = g2 = 1.0

    for i in range(iterations):
        u = torch.randn(theta.size()).type(torch.FloatTensor)
        g1, count1 = backtracking_line_search(model, x0, y0, theta + beta * u, initial_lbd = g1)
        g2, count2 = backtracking_line_search(model, x0, y0, theta, initial_lbd = g2)
        gradient = (g1-g2)/beta * u
        if (i+1)%100 == 0:
            target = model.module.predict(x0 + g2 * theta)
            print("Iteration %3d: g(theta + beta*u) %.4f g(theta) %.4f distance %.4f target %d" % (i+1, g1, g2, torch.norm(g2*theta), target))
        theta.sub_(alpha*gradient)
        query_count += count1 + count2 + 1

    timeend = time.time()
    print("Number of queries: %d, time %.4f seconds" % (query_count, timeend-timestart))
    return x0 + g2*theta

def backtracking_line_search(model, x0, y0, theta, initial_lbd = 1.0):
    query_count = 0
    lbd = initial_lbd
    while model.module.predict(x0 + lbd*theta) == y0:
        lbd *= 2.0
        query_count += 1

    while model.module.predict(x0 + lbd*theta) != y0:
        lbd = lbd/2.0
        query_count += 1
    
    lbd_star = lbd*2.0

    # fine-grained search 
    for lbd in np.linspace(lbd_star, 0.0, 200):
        if lbd < lbd_star and model.module.predict(x0 + lbd*theta) != y0:
            lbd_star = lbd
        query_count += 1

    return lbd_star, query_count


if __name__ == '__main__':
    train_loader, test_loader, train_dataset, test_dataset = load_data()
    net = CNN()
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #train(net, train_loader)
    load_model(net, 'models/cifar10_gpu.pt')
    test(net, test_loader)
    #save_model(net,'./cifar10.pt')
    num_images = 5
    for _ in range(num_images):
        idx = random.randint(1, 10000)
        image, label = test_dataset[idx]
        print("Original label: ", label)
        print("Predicted label: ", net.module.predict(image))
        adversarial = attack(net, train_dataset, image, label)
        print("Predicted label: ", net.module.predict(adversarial))
        #print("mindist: ", mindist)
        #print(theta)





