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

alpha = 0.2
beta = 0.001

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
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, train_dataset, test_dataset


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        image = Variable(image).view(1,1,28,28)
        if torch.cuda.is_available():
            image = image.cuda()
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

def test(model, test_loader):
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

def save_model(model, filename):
    """ Save the trained model """
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    """ Load the training model """
    model.load_state_dict(torch.load(filename))

def attack(model, train_dataset, x0, y0, alpha = 0.018, beta = 0.05, query_limit = 100000):
    """ Attack the original image and return adversarial example"""

    if (model.predict(x0) != y0):
        print("Fail to classify the image. No need to attack.")
        return x0

    num_samples = 1000 
    best_theta = None
    best_distortion = float('inf')
    g_theta = None
    query_search_each = 100
    query_count = 0
    print("Searching for the initial direction on %d samples: " % (num_samples))

    timestart = time.time()
    samples = set(random.sample(range(len(train_dataset)), num_samples))
    for i, (xi, yi) in enumerate(train_dataset):
        if i not in samples:
            continue
        query_count += 1
        if model.predict(xi) != y0:
            theta = xi - x0
            #query_count += query_search_each
            lbd, count = fine_grained_binary_search(model, x0, y0, theta, query_limit = query_search_each)
            query_count += count
            distortion = torch.norm(lbd*theta)
            if distortion < best_distortion:
                best_theta, g_theta = theta, lbd
                best_distortion = distortion
                print("--------> Found distortion %.4f and g_theta = %.4f" % (best_distortion, g_theta))

    timeend = time.time()
    print("==========> Found best distortion %.4f and g_theta = %.4f in %.4f seconds using %d queries" % (best_distortion, g_theta, timeend-timestart, query_count))

    query_limit -= query_count

  
    timestart = time.time()

    query_search_each = 200  # limit for each lambda search
#    iterations = (query_limit - query_search_each)//(2*query_search_each)
    iterations = 5000
    g1 = 1.0
    g2 = g_theta
    theta = best_theta

    opt_count = 0
    for i in range(iterations):
        u = torch.randn(theta.size()).type(torch.FloatTensor)
        u = u/torch.norm(u)
        #g1, count = fine_grained_binary_search(model, x0, y0, theta + beta * u, initial_lbd = g1, query_limit = query_search_each)
        #opt_count += count
        g2, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd = g2, query_limit = query_search_each)
        opt_count += count
        ttt = theta+beta * u
        ttt = ttt/torch.norm(ttt)
        g1, count = fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2, query_limit = query_search_each)
        opt_count += count
        if g1 == float('inf'):
            print("WHY g1 = INF???")
        if g2 == float('inf'):
            print("WHY g2 = INF???")
        if (i+1)%50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))
        gradient = (g1-g2)/torch.norm(ttt-theta) * u
        #gradient = (g1-g2)/beta * u
        theta.sub_(alpha*gradient)
        theta = theta/torch.norm(theta)

    g2, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd = g2, query_limit = query_search_each)
    distortion = torch.norm(g2*theta)
    target = model.predict(x0 + g2*theta)
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d \nTime: %.4f seconds" % (distortion, target, timeend-timestart))
    return x0 + g2*theta

def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = 1.0, query_limit = 200):
    nquery = 0
    lbd = initial_lbd
   
    if model.predict(x0+lbd*theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while model.predict(x0+lbd_hi*theta) == y0:
            lbd_hi = lbd_hi*1.01
            nquery += 1
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while model.predict(x0+lbd_lo*theta) != y0 :
            lbd_lo = lbd_lo*0.99
            nquery += 1

    while (lbd_hi - lbd_lo) > 1e-8:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery


def fine_grained_binary_search(model, x0, y0, theta, initial_lbd = 1.0, query_limit = 200):
    nquery = 0
    lbd = initial_lbd
    while model.predict(x0 + lbd*theta) == y0:
        lbd *= 2.0
        query_limit -= 1

    if lbd > 1000 or query_limit < 0:
        print("WHY lbd > 1000")
        return float('inf')

    # fine-grained search 
    query_fine_grained = query_limit // 2
    query_binary_search = query_limit - query_fine_grained

    lambdas = np.linspace(0.0, lbd, query_fine_grained)[1:]
    #print lambdas
    lbd_hi = lbd
    lbd_hi_index = 0
    for i, lbd in enumerate(lambdas):
        nquery += 1
        if model.predict(x0 + lbd*theta) != y0:
            lbd_hi = lbd
            lbd_hi_index = i
            break
    #print lbd_hi, lbd_hi_index
    lbd_lo = lambdas[lbd_hi_index - 1]

    while query_binary_search > 0:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
        query_binary_search -= 1
        if (lbd_hi - lbd_lo) < 1e-7:
            break
    return lbd_hi, nquery

def main():
    train_loader, test_loader, train_dataset, test_dataset = load_data()
    net = CNN()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        
    #train(net, train_loader)
    #load_model(net, 'models/mnist_gpu.pt')
    load_model(net, 'models/mnist.pt')
    test(net, test_loader)
    #save_model(net,'./models/mnist_gpu.pt')
    #save_model(net,'./models/mnist.pt')
    net.eval()

    model = net.module if torch.cuda.is_available() else net

    query_limit = 100000
    num_images = 50

    for i, (image, label) in enumerate(test_dataset):
        if i >= num_images:
            break
        print("\n\n\n\n======== Image %d =========" % i)
        show(image.numpy())
        print("Original label: ", label)
        print("Predicted label: ", model.predict(image))
        
        adversarial = attack(model, train_dataset, image, label, alpha = alpha, beta = beta, query_limit = query_limit)
        show(adversarial.numpy())
        print("Predicted label for adversarial example: ", model.predict(adversarial))
        #print("mindist: ", mindist)
        #print(theta)

    print("\n\n\n\n\n Random Sample\n\n\n")

    for _ in range(num_images):
        idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        show(image.numpy())
        print("Original label: ", label)
        print("Predicted label: ", model.predict(image))
        
        adversarial = attack(model, train_dataset, image, label, alpha = alpha, beta = beta, query_limit = query_limit)
        show(adversarial.numpy())
        print("Predicted label for adversarial example: ", model.predict(adversarial))


if __name__ == '__main__':
    timestart = time.time()
    main()
    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))

    # estimate time per one iteration (two examples)
    # query = 100000 -> 100 seconds 
    # query = 200000 
    # query = 500000 ->  
    # query = 1000000 ->  
