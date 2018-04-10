import time, sys
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from models import MNIST, CIFAR10, IMAGENET, SimpleMNIST, load_mnist_data, load_cifar10_data, imagenettest, load_model, show_image

alpha = 0.2
beta = 0.001


def attack_untargeted(model, train_loader, x0, y0, alpha = 0.2, beta = 0.001, iterations= 200000):
    """ Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
    """

    if (model.predict(x0) != y0):
        print("Fail to classify the image. No need to attack.")
        return x0

    num_samples = 1000 
    best_theta = None
    best_distortion = float('inf')
    g_theta = None
    query_count = 0
    timestart = time.time()
    b_train_size = 1000
    b_best_lbd = float('inf')
    dim1 = x0.size()[0]
    dim3 = x0.size()[2]
    for i, (xi, yi) in enumerate(train_loader):
        if i == 1:
            break
        xi,yi=xi.cuda(),yi.cuda()
        temp_x0, temp_y0 = x0, y0
        predicted = model.predict_batch(xi)
        b_index = (predicted !=y0).nonzero().squeeze()
        if len(b_index.size()) == 0:
            continue
        xi = xi[b_index]
        temp_x0 = temp_x0.expand(xi.size()).cuda()
        theta = xi - temp_x0
        initial_lbd = torch.norm(torch.norm(torch.norm(theta,2,1),2,1),2,1)
        initial_lbd = initial_lbd.unsqueeze(1).unsqueeze(2).expand(xi.size()[0],dim1,dim3).unsqueeze(3).expand(xi.size()[0],dim1,dim3,dim3)
        theta /= initial_lbd
        lbd, query_count = initial_fine_grained_binary_search(model, temp_x0, y0, theta)
        best_lbd, best_index = torch.min(lbd,0)
        best_theta = theta[best_index]
        if best_lbd[0] < b_best_lbd:
            b_best_lbd = best_lbd[0]
            b_best_theta = best_theta.clone()
            print("--------> Found g() %.4f" %b_best_lbd)

    best_theta, g_theta = b_best_theta.cpu(), b_best_lbd
    timeend = time.time()
    print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (b_best_lbd, timeend-timestart, query_count))

    #query_limit -= query_count

     
    timestart = time.time()

    #query_search_each = 200  # limit for each lambda search
    #iterations = (query_limit - query_search_each)//(2*query_search_each)
    iterations = 200000
    g1 = 1.0
    g2 = g_theta
    theta = best_theta
    now_o = g2*theta
    delta = 0.01
    epsilon = 0.001

    opt_count = 0
    success_count = 0
    n_adjust = 1000
    torch.manual_seed(0)
    for i in range(iterations):
        u = torch.randn(theta.size()).type(torch.FloatTensor)
        new_o = now_o + u*delta
        new_o = new_o *( torch.norm(now_o) / torch.norm(new_o))
        if model.predict(new_o) != y0:
            success_count += 1
        new_o = new_o - epsilon*(new_o)/torch.norm(new_o)
        if model.predict(x0+new_o) != y0:
            now_o = new_o
        """ ## I found adaptive step size is not good....
        if (i+1)%n_adjust == 0:
            ratio = float(success_count)/float(n_adjust)
            print ratio, delta, epsilon
            if ratio < 0.2:
                delta = delta / 1.5
                epsilon = epsilon /1.5
            elif ratio > 0.5:
                delta = delta * 1.5
                epsilon = epsilon * 1.5
        """
        if (i+1)%5000 == 0:
            print("Iteration %3d distortion %.4f query %d" % (i+1, torch.norm(now_o), (i+1)*2))

    distortion = torch.norm(now_o)
    target = model.predict(now_o)
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (distortion, target, query_count + iterations, timeend-timestart))
    return x0+now_o

def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = 1.0):
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

def initial_fine_grained_binary_search(model, x0, y0, theta, initial_lbd = 1.0):
    dim1 = x0.size()[1]
    dim3 = x0.size()[2] 
    nquery = 0
    initial_lbd = torch.ones(theta.size()).cuda()
    lbd = initial_lbd
    limit = torch.ones(lbd.size()).cuda()
    predicted = model.predict_batch(x0+ lbd * theta)
    nquery += 10
    candidate = (predicted == y0).nonzero().view(-1)
    while len(candidate.size())>0:
        lbd[candidate] = lbd[candidate].mul(1.05)
        nquery += candidate.size()[0]
        limit.resize_(candidate.size())
        if torch.max(lbd) > 100: 
            break
        predicted = model.predict_batch(x0+ lbd * theta)
        nquery += candidate.size()[0]
        candidate = (predicted == y0).nonzero().view(-1)
    num_intervals = 100
    
    lambdas = torch.randn(lbd.size()[0],num_intervals-1).cuda()
    for i in range(lbd.size()[0]):
        lambdas_t = np.linspace(0.0, lbd[i][0][0][0], num_intervals)[1:]
        lambdas_t = torch.from_numpy(lambdas_t).type(torch.FloatTensor)
        lambdas[i] = lambdas_t
    lbd_hi, lbd_hi_index = torch.max(lambdas,1)
    lbd_lo = lbd_hi.clone()
    for i in range(lbd.size()[0]):
        temp_lbd = lambdas[i].unsqueeze(1).unsqueeze(2).expand(num_intervals-1,dim1,dim3).unsqueeze(3).expand(num_intervals-1,dim1,dim3,dim3)
        temp_theta = theta[i].unsqueeze(0).expand(num_intervals-1,dim1,dim3,dim3)
        temp_x0 = x0[i].unsqueeze(0).expand(num_intervals-1,dim1,dim3,dim3)
        predicted = model.predict_batch(temp_x0+temp_lbd*temp_theta)
        candidate = (predicted!=y0).nonzero().view(-1)
        nquery += num_intervals-1 - candidate.size()[0]
        if len(candidate.size())==0:
            lbd_hi_index[i] = 0
            #print(lbd[i][0],lbd_hi[i])
            lbd_hi[i] = lbd[i][0][0][0]
            #return float('inf'), nquery
        else:
            lbd_hi_index[i] = torch.min(candidate)
            lbd_hi[i] = lambdas[i][lbd_hi_index[i]]
        lbd_lo[i] = lambdas[i][lbd_hi_index[i] - 1]

    while torch.max(lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        temp_lbd_mid = lbd_mid.unsqueeze(1).unsqueeze(2).expand(lbd_mid.size()[0],dim1,dim3).unsqueeze(3).expand(lbd_mid.size()[0],dim1,dim3,dim3)
        predicted = model.predict_batch(x0+temp_lbd_mid*theta)
        nquery += lbd_mid.size()[0]
        
        candidate_y = (predicted!=y0).nonzero().view(-1)
        candidate_n = (predicted==y0).nonzero().view(-1)
        if len(candidate_y.size())>0:
            lbd_hi[candidate_y] = lbd_mid[candidate_y]
        if len(candidate_n.size())>0:
            lbd_lo[candidate_n] = lbd_mid[candidate_n]
    return lbd_hi, nquery


def fine_grained_binary_search(model, x0, y0, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd
    while model.predict(x0 + lbd*theta) == y0:
        lbd *= 2.0
        nquery += 1

    num_intervals = 100

    lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
    lbd_hi = lbd
    lbd_hi_index = 0
    for i, lbd in enumerate(lambdas):
        nquery += 1
        if model.predict(x0 + lbd*theta) != y0:
            lbd_hi = lbd
            lbd_hi_index = i
            break

    lbd_lo = lambdas[lbd_hi_index - 1]

    while (lbd_hi - lbd_lo) > 1e-7:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def attack_single(model, train_loader, image, label, target = None, alpha=0.2):
    #show_image(image.numpy())
    print("Original label: ", label)
    print("Predicted label: ", model.predict(image))
    if target == None:
        adversarial = attack_untargeted(model, train_loader, image, label, alpha = alpha, beta = beta, iterations = 200000)
    else:
        print("Targeted attack: %d" % target)
        adversarial = attack_targeted(model, train_loader, image, label, target, alpha = alpha, beta = beta, iterations = 200000)
    show_image(adversarial.numpy())
    print("Predicted label for adversarial example: ", model.predict(adversarial))
    return torch.norm(adversarial - image)



def attack_mnist(alpha):
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    net = MNIST()
    #train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()
    #net = CIFAR10()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        
    load_model(net, 'models/mnist_gpu.pt')
    #load_model(net, 'models/cifar10.pt')
    net.eval()

    model = net.module if torch.cuda.is_available() else net

    num_images = 10
    
    print("\n\n\n\n\n Running on {} random images \n\n\n".format(num_images))
    distortion_random_sample = 0.0

    random.seed(0)
    for _ in range(num_images):
        idx = random.randint(100, len(test_dataset)-1)
        #idx = 3743
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        targets = list(range(10))
        targets.pop(label)
        target = random.choice(targets)
        #target = 4
        #target = None   #--> uncomment of untarget
        distortion_random_sample += attack_single(model, train_loader, image, label, target, alpha)

    #print("\n\n\n\n\n Running on first {} images \n\n\n".format(num_images))
    print("Average distortion on random {} images is {}".format(num_images, distortion_random_sample/num_images))
    '''
    distortion_fix_sample = 0.0

    for i, (image, label) in enumerate(test_loader):
        #targets = list(range(10))
        #targets.pop(label)
        #target = random.choice(targets)
        #target = None   --> uncomment of untarget
        distortion_fix_sample += attack_mnist_single(model, train_loader, image, label)

    print("\n\nAverage distortion on first {} images is {}".format(num_images, distortion_fix_sample/num_images))
    print("Average distortion on random {} images is {}".format(num_images, distortion_random_sample/num_images))
    '''
def attack_cifar(alpha):
    #train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    #net = MNIST()
    train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()
    net = CIFAR10()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        
    #load_model(net, 'models/mnist_gpu.pt')
    load_model(net, 'models/cifar10_gpu.pt')
    net.eval()

    model = net.module if torch.cuda.is_available() else net

    num_images = 100
    
    print("\n\n\n\n\n Running on {} random images \n\n\n".format(num_images))
    distortion_random_sample = 0.0

    random.seed(0)
    for _ in range(num_images):
        idx = random.randint(100, len(test_dataset)-1)
        #idx = 5474
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        targets = list(range(10))
        targets.pop(label)
        target = random.choice(targets)
        #target = 3
        target = None   #--> uncomment of untarget
        distortion_random_sample += attack_single(model, train_loader, image, label, target, alpha)

    #print("\n\n\n\n\n Running on first {} images \n\n\n".format(num_images))
    print("Average distortion on random {} images is {}".format(num_images, distortion_random_sample/num_images))
    '''
    distortion_fix_sample = 0.0

    for i, (image, label) in enumerate(test_loader):
        #targets = list(range(10))
        #targets.pop(label)
        #target = random.choice(targets)
        #target = None   --> uncomment of untarget
        distortion_fix_sample += attack_mnist_single(model, train_loader, image, label)

    print("\n\nAverage distortion on first {} images is {}".format(num_images, distortion_fix_sample/num_images))
    print("Average distortion on random {} images is {}".format(num_images, distortion_random_sample/num_images))
    '''
def attack_imgnet(alpha):
    '''
    model_name = 'inceptionresnetv2' # could be fbresnet152 or inceptionresnetv2
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    '''
    #model = torchvision.models.vgg19(pretrained=True)
    model = IMAGENET('vgg19')
    
    #model = model.module if torch.cuda.is_available() else net
    '''
    input_size = model.input_size
    input_space = model.input_space
    input_range = model.input_range
    mean = model.mean
    std = model.std
    scale = 0.875
    tfs = []
    tfs.append(transforms.Resize(int(math.floor(max(input_size)/scale))))
    tfs.append(transforms.CenterCrop(max(input_size)))
    tfs.append(transforms.ToTensor())
    tfs.append(ToSpaceBGR(input_space=='BGR'))
    tfs.append(ToRange255(max(input_range)==255))
    tfs.append(transforms.Normalize(mean=mean, std=std))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    tfs = []
    tfs.append(transforms.Resize(256))
    tfs.append(transforms.CenterCrop(224))
    tfs.append(transforms.ToTensor())
    tfs.append(normalize)
    tfs = transforms.Compose(tfs)
    test_dataset = ImagenetTestDataset('/data/test', tfs)
    
    #print(test_dataset[0][0])
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)
    #for i, (image, label) in enumerate(test_loader):
    #    output = model_predict(model, image)
    #    print(output)
    '''
    test_loader, test_dataset = imagenettest()
    distortion_random_sample = 0
    num_images = 10
    random.seed(0)
    for _ in range(num_images):
        idx = random.randint(100, len(test_dataset)-1)
        #idx = 3743
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        targets = list(range(1000))
        targets.pop(label)
        target = random.choice(targets)
        #target = 4
        target = None   #--> uncomment of untarget
        distortion_random_sample += attack_single(model, test_loader, image, label, target, 0.2)

    #print("\n\n\n\n\n Running on first {} images \n\n\n".format(num_images))
    print("Average distortion on random {} images is {}".format(num_images, distortion_random_sample/num_images))
    '''
    distortion_fix_sample = 0.0

    for i, (image, label) in enumerate(test_loader):
        #targets = list(range(10))
        #targets.pop(label)
        #target = random.choice(targets)
        #target = None   --> uncomment of untarget
        distortion_fix_sample += attack_single(model, train_loader, image, label)

    print("\n\nAverage distortion on first {} images is {}".format(num_images, distortion_fix_sample/num_images))
    print("Average distortion on random {} images is {}".format(num_images, distortion_random_sample/num_images))
    '''




if __name__ == '__main__':
    timestart = time.time()
    alpha = float(sys.argv[1])
    #attack_mnist(alpha)
    attack_cifar(alpha)
    #attack_imgnet(alpha)
    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
