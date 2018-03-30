import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from models import MNIST, CIFAR10, load_mnist_data, load_cifar10_data, load_model, show_image

alpha = 0.2
beta = 0.001

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
    #old_val = np.maximum(np.minimum(old_val, 1.0), 0.0)    
# print(grad)
    # print(old_val - m[indice])
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1


def attack(input, label, net, c, batch_size= 128, TARGETED=False):
    input_v = Variable(input.cuda())
    n_class = 10
    index = label.view(-1,1)
    label_onehot = torch.FloatTensor(input_v.size()[0] , n_class)
    label_onehot.zero_()
    label_onehot.scatter_(1,index,1)
    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
	#print(label_onehot.scatter)
    var_size = input_v.view(-1).size()[0]
    #print(var_size)
    real_modifier = torch.FloatTensor(input_v.size()).zero_().cuda()
    for iter in range(200): 
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
            output = net(torch.clamp(input_v + modifier_v,0,1))
            #print(output)
            real = torch.max(torch.mul(output, label_onehot_v), 1)[0]
            other = torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0]
            loss1 = torch.sum(modifier_v*modifier_v)/1
            if TARGETED:
                loss2 = c* torch.sum(torch.clamp(other - real, min=0))
            else:
                loss2 = c* torch.sum(torch.clamp(real - other, min=0))
            error = loss2 + loss1 
            #error = loss2
            losses[i] = error.data[0]
        if (iter+1)%1 == 0:
            print(np.sum(losses))
        #if loss2.data[0]==0:
        #    break
        grad = np.zeros(batch_size, dtype=np.float32)
        mt = np.zeros(var_size, dtype=np.float32)
        vt = np.zeros(var_size, dtype=np.float32)
        adam_epoch = np.ones(var_size, dtype = np.int32)
        np_modifier = real_modifier.cpu().numpy()
        lr = 0.1
        beta1, beta2 = 0.9, 0.999
        #for i in range(1):
        #print(np.count_nonzero(np_modifier))
        coordinate_ADAM(losses, random_set[:batch_size], grad, batch_size, mt, vt, np_modifier, lr, adam_epoch, beta1, beta2)
        real_modifier = torch.from_numpy(np_modifier)
    real_modifier_v = Variable(real_modifier, requires_grad=True).cuda()
    print(torch.norm(real_modifier_v)) 
    return (input_v + real_modifier_v).data.cpu()


def zoo_attack(dataset):
    if dataset == 'cifar10':
        train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()
        net = CIFAR10()
    else:
        train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
        net = MNIST()

    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
    
    if dataset == 'cifar10':
        load_model(net, 'models/cifar10_gpu.pt')
    else:
        load_model(net, 'models/mnist_gpu.pt')
    #save_model(net,'./models/mnist.pt')
    net.eval()

    model = net.module

    #num_images = 10
    test_dataset = dsets.MNIST(root='./data/mnist', train=True, transform=transforms.ToTensor(), download=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    for i, (image, label) in enumerate(test_loader):
        #print("\n\n\n\n======== Image %d =========" % i)
        #show_image(image.numpy())
        print("Original label:" , label)
        print("Predicted label:" , model.predict_batch(image))
        adversarial = attack(image, label, model, 1)
        print("Predicted label for adversarial example: ", model.predict_batch(adversarial))
        #print("mindist: ", mindist)
        #print(theta)
    '''
    print("\n\n\n\n\n Random Sample\n\n\n")

    for _ in range(num_images):
        idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        show_image(image.numpy())
        print("Original label: ", label)
        print("Predicted label: ", net.module.predict(image))
        adversarial = attack(image, label, net.module, 1)
        show_image(adversarial.numpy())
        print("Predicted label for adversarial example: ", net.module.predict(adversarial))
    '''

if __name__ == '__main__':
    timestart = time.time()
    zoo_attack('mnist')
    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))

    # estimate time per one iteration (two examples)
    # query = 100000 -> 100 seconds 
    # query = 200000 
    # query = 500000 ->  
    # query = 1000000 ->  
