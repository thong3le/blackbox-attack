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
    train_dataset = dsets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
    #transform_train = tfs.Compose([
    #    tfs.RandomCrop(32, padding=4),
    #    tfs.RandomHorizontalFlip(),
    #    tfs.ToTensor()
    #    ])
 
    #train_dataset = dsets.CIFAR10('./cifar10-py', download=False, train=True, transform= transforms.ToTensor())
    #test_dataset = dsets.CIFAR10('./cifar10-py', download=False, train=False, transform= transforms.ToTensor())

    
    
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
        image = Variable(image).cuda().view(1,1,28,28)
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

    print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct / total))

def save_model(model, filename):
    """ Save the trained model """
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    """ Load the training model """
    model.load_state_dict(torch.load(filename))

def coordinate_ADAM(losses, indice, grad,  batch_size, mt_arr, vt_arr, real_modifier, lr, adam_epoch, beta1, beta2):
    # indice = np.array(range(0, 3*299*299), dtype = np.int32)
    for i in range(batch_size):
        grad[i] = (losses[i*2] - losses[i*2+1]) / 0.0002 
    # true_grads = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
    # true_grads, losses, l2s, scores, nimgs = self.sess.run([self.grad_op, self.loss, self.l2dist, self.output, self.newimg], feed_dict={self.modifier: self.real_modifier})
    # grad = true_grads[0].reshape(-1)[indice]
    # print(grad, true_grads[0].reshape(-1)[indice])
    # self.real_modifier.reshape(-1)[indice] -= self.LEARNING_RATE * grad
    # self.real_modifier -= self.LEARNING_RATE * true_grads[0]
    # ADAM update
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1 - beta2) * (grad * grad)
    vt_arr[indice] = vt
    # epoch is an array; for each index we can have a different epoch number
    epoch = adam_epoch[indice]
    corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
    m = real_modifier.reshape(-1)
    old_val = m[indice] 
    old_val -= lr * corr * mt / (np.sqrt(vt) + 1e-8)
    # print(grad)
    # print(old_val - m[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1


def attack(input, label, net, c, batch_size= 1, TARGETED=False):
    input_v = Variable(input.cuda()).view(1,1,28,28)
    n_class = 10
    index = torch.LongTensor([label]).view(-1,1)
    label_onehot = torch.FloatTensor(input_v.size()[0] , n_class)
    label_onehot.zero_()
    label_onehot.scatter_(1,index,1)
    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
	#print(label_onehot.scatter)
    var_size = input_v.view(-1).size()[0]
    #print(var_size)
    real_modifier = torch.FloatTensor(input_v.size()).zero_().cuda()
    for iter in range(2000): 
        random_set = np.random.permutation(var_size)
        losses = np.zeros(2*batch_size, dtype=np.float32)
        #print(torch.sum(real_modifier))
        for i in range(2*batch_size):
            modifier = real_modifier.clone().view(-1)
            if i%2==0:
                modifier[random_set[i//2]] += 0.0001 
            else:
                modifier[random_set[i//2]] -= 0.0001
            modifier = modifier.view(input_v.size())
            modifier_v = Variable(modifier, requires_grad=True).cuda()
            output = net(input_v + modifier_v)
            #print(output)
            real = torch.max(torch.mul(output, label_onehot_v), 1)[0]
            other = torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0]
            loss1 = torch.sum(modifier_v*modifier_v)
            if TARGETED:
                loss2 = c* torch.sum(torch.clamp(other - real, min=0))
            else:
                loss2 = c* torch.sum(torch.clamp(real - other, min=0))
            error = loss1 + loss2
            losses[i] = error.data[0]
        if (iter+1)%50 == 0:
            print(np.sum(losses))
        #if loss2.data[0]==0:
        #    break
        grad = np.zeros(batch_size, dtype=np.float32)
        mt = np.zeros(var_size, dtype=np.float32)
        vt = np.zeros(var_size, dtype=np.float32)
        adam_epoch = np.ones(var_size, dtype = np.int32)
        np_modifier = real_modifier.cpu().numpy()
        lr = 0.2
        beta1, beta2 = 0.9, 0.999
        #for i in range(batch_size):
        coordinate_ADAM(losses, random_set[:batch_size], grad, batch_size, mt, vt, np_modifier, lr, adam_epoch, beta1, beta2)
        real_modifier = torch.from_numpy(np_modifier)
    real_modifier_v = Variable(real_modifier, requires_grad=True).cuda()
    
    return (input_v + real_modifier_v).data.cpu()


def main():
    train_loader, test_loader, train_dataset, test_dataset = load_data()
    net = CNN()
    net.cuda()
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net = torch.nn.DataParallel(net, device_ids=[0])
    #train(net, train_loader)
    load_model(net, 'models/mnist_gpu.pt')
    test(net, test_loader)
    #save_model(net,'./models/mnist.pt')
    net.eval()


    query_limit = 100000
    num_images = 50

    for i, (image, label) in enumerate(test_dataset):
        if i >= num_images:
            break
        print("\n\n\n\n======== Image %d =========" % i)
        show(image.numpy())
        print("Original label: ", label)
        print("Predicted label: ", net.module.predict(image))
        adversarial = attack(image, label, net.module, 1)
        show(adversarial.numpy())
        print("Predicted label for adversarial example: ", net.module.predict(adversarial))
        #print("mindist: ", mindist)
        #print(theta)

    print("\n\n\n\n\n Random Sample\n\n\n")

    for _ in range(num_images):
        idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        show(image.numpy())
        print("Original label: ", label)
        print("Predicted label: ", net.module.predict(image))
        adversarial = attack(image, label, net.module, 1)
        show(adversarial.numpy())
        print("Predicted label for adversarial example: ", net.module.predict(adversarial))



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
