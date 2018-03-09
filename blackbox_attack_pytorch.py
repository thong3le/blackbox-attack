import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

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
    train_dataset = dsets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset


class CNN(nn.Module):
    """ Custom CNN for MNIST
        stride = 1, padding = 2
        Layer 1: Conv2d 5x5x16, BatchNorm(16), ReLU, Max Pooling 2x2
        Layer 2: Conv2d 5x5x32, BatchNorm(32), ReLU, Max Pooling 2x2
        FC 10
    """
    def __init__(self):
        super(CNN, self).__init__()
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

   

def train(model, train_loader):
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

def test(model, test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

def save_model(model, filename):
    """ Save the trained model """
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    """ Load the training model """
    model.load_state_dict(torch.load(filename))

def attack(model, dataset, x0, y0, alpha = 0.018, beta = 0.05, iterations = 1000):

    if (model.predict(x0) != y0):
        print("Fail to classify the image. No need to attack.")
        return

    timestart = time.time()
    print("Looking for the initial direction: ")
    mindist = float('inf')
    theta = None
    info = None
    for i, (xi, yi) in enumerate(dataset):
        diff = xi-x0
        dist = torch.norm(xi-x0)
        if random.random() < 0.1 and model.predict(xi) != y0 and dist < mindist:
            mindist = dist
            theta, info = diff, (i, yi)

    timeend = time.time()
    print("-----> Found image indexing at %d labelled %d within distance %.4f from original image in %.4f seconds" % (info[0], info[1], mindist, timeend-timestart))

    timestart = time.time()

    query_count = 0
    g1 = g2 = 1.0

    for i in range(iterations):
        u = torch.randn(theta.size())
        g1, count1 = backtracking_line_search(model, x0, y0, theta + beta * u, initial_lbd = g1)
        g2, count2 = backtracking_line_search(model, x0, y0, theta, initial_lbd = g2)
        gradient = (g1-g2)/beta * u
        if (i+1)%100 == 0:
            target = model.predict(x0 + g2 * theta)
            print("Iteration %3d: g(theta + beta*u) %.4f g(theta) %.4f distance %.4f target %d" % (i+1, g1, g2, torch.norm(g2*theta), target))
        theta.sub_(alpha*gradient)
        query_count += count1 + count2 + 1

    timeend = time.time()
    print("Number of queries: %d, time %.4f seconds" % (query_count, timeend-timestart))
    return x0 + g2*theta

def backtracking_line_search(model, x0, y0, theta, initial_lbd = 1.0):
    query_count = 0
    lbd = initial_lbd
    while model.predict(x0 + lbd*theta) == y0:
        lbd *= 2.0
        query_count += 1

    while model.predict(x0 + lbd*theta) != y0:
        lbd = lbd/2.0
        query_count += 1
    
    lbd_star = lbd*2.0

    # fine-grained search 
    for lbd in np.linspace(lbd_star, 0.0, 200):
        if lbd < lbd_star and model.predict(x0 + lbd*theta) != y0:
            lbd_star = lbd
        query_count += 1

    return lbd_star, query_count


if __name__ == '__main__':
    train_loader, test_loader, train_dataset, test_dataset = load_data()
    cnn = CNN()
    #train(cnn, train_loader)
    load_model(cnn, 'models/simple_mnist_cnn_pt.pkl')
    #test(cnn, test_loader)

    num_images = 5
    for _ in range(num_images):
        idx = random.randint(1, 10000)
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        show(image.numpy())
        print("Original label: ", label)
        print("Predicted label: ", cnn.predict(image))
        adversarial = attack(cnn, train_dataset, image, label)
        show(adversarial.numpy())
        print("Predicted label: ", cnn.predict(adversarial))
        #print("mindist: ", mindist)
        #print(theta)





