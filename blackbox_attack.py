import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from models import MNIST, CIFAR10, SimpleMNIST, load_mnist_data, load_cifar10_data, load_model, show_image

alpha = 0.2
beta = 0.001

def attack_targeted(model, train_dataset, x0, y0, target, alpha = 0.1, beta = 0.001, iterations = 1000):
    """ Attack the original image and return adversarial example of target t
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        t: target
    """

    if (model.predict(x0) != y0):
        print("Fail to classify the image. No need to attack.")
        return x0

    # STEP I: find initial direction (theta, g_theta)

    num_samples = 10000 
    best_theta, g_theta = None, float('inf')
    query_count = 0

    print("Searching for the initial direction on %d samples: " % (num_samples))
    timestart = time.time()
    samples = set(random.sample(range(len(train_dataset)), num_samples))
    for i, (xi, yi) in enumerate(train_dataset):
        if i not in samples:
            continue
        query_count += 1
        if model.predict(xi) == target:
            theta = xi - x0
            initial_lbd = torch.norm(theta)
            theta = theta/torch.norm(theta)
            lbd, count = fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print("--------> Found distortion %.4f" % g_theta)

    timeend = time.time()
    print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))


    # STEP II: seach for optimal
    timestart = time.time()
    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta

    opt_count = 0
    for i in range(iterations):

        while beta > 1e-6:
            u = torch.randn(theta.size())
            u = u/torch.norm(u)
            ttt = theta+beta * u
            ttt = ttt/torch.norm(ttt)
            ttt = ttt.type(torch.FloatTensor)
            g1, count = fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt, initial_lbd = g2)
            opt_count += count
            if g1 != float('inf'):
                break
            beta *= 0.8

        if (i+1)%10 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d alpha %.5f beta %.5f" % (i+1, g1, g2, g2, opt_count, alpha, beta))
            
        gradient = (g1-g2)/torch.norm(ttt-theta) * u
        
        # make sure the step-size is good
        new_alpha = alpha
        while new_alpha > 1e-6:
            new_theta = theta - new_alpha*gradient
            new_theta = new_theta/torch.norm(new_theta)
            new_g2, count = fine_grained_binary_search_targeted(model, x0, y0, target, new_theta, initial_lbd = g2)
            opt_count += count
            if new_g2 != float('inf'):
                g2 = new_g2
                theta = new_theta
                break
            new_alpha *= 0.8

        if beta <= 1e-6 or new_alpha <= 1e-6:
            break

        if g2 < g_theta:
            best_theta, g_theta = theta.clone(), g2

    out_target = model.predict(x0 + best_theta*g_theta)  # should be the target
    timeend = time.time()
    print("\nAdversarial Example Tageted %d Found Successfully: distortion %.4f target %d queries %d alpha %.5f beta %.5f \nTime: %.4f seconds" % (target, g_theta, out_target, query_count + opt_count, alpha, beta, timeend-timestart))
    return x0 + g_theta*best_theta

def fine_grained_binary_search_local_targeted(model, x0, y0, t, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd
   
    if model.predict(x0+lbd*theta) != t:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while model.predict(x0+lbd_hi*theta) != t:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi > 100: 
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while model.predict(x0+lbd_lo*theta) == t:
            lbd_lo = lbd_lo*0.99
            nquery += 1

    while (lbd_hi - lbd_lo) > 1e-8:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def fine_grained_binary_search_targeted(model, x0, y0, t, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd

    while model.predict(x0 + lbd*theta) != t:
        lbd *= 1.05
        nquery += 1
        if lbd > 100: 
            return float('inf'), nquery

    num_intervals = 100

    lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
    lbd_hi = lbd
    lbd_hi_index = 0
    for i, lbd in enumerate(lambdas):
        nquery += 1
        if model.predict(x0 + lbd*theta) == t:
            lbd_hi = lbd
            lbd_hi_index = i
            break

    lbd_lo = lambdas[lbd_hi_index - 1]

    while (lbd_hi - lbd_lo) > 1e-7:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid

    return lbd_hi, nquery



def attack_untargeted(model, train_dataset, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000):
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

    print("Searching for the initial direction on %d samples: " % (num_samples))
    timestart = time.time()
    samples = set(random.sample(range(len(train_dataset)), num_samples))
    for i, (xi, yi) in enumerate(train_dataset):
        if i not in samples:
            continue
        query_count += 1
        if model.predict(xi) != y0:
            theta = xi - x0
            theta = theta/torch.norm(theta)
            lbd, count = fine_grained_binary_search(model, x0, y0, theta)
            query_count += count
            distortion = torch.norm(lbd*theta)
            if distortion < best_distortion:
                best_theta, g_theta = theta, lbd
                best_distortion = distortion
                print("--------> Found distortion %.4f and g_theta = %.4f" % (best_distortion, g_theta))

    timeend = time.time()
    print("==========> Found best distortion %.4f and g_theta = %.4f in %.4f seconds using %d queries" % (best_distortion, g_theta, timeend-timestart, query_count))
  
    timestart = time.time()
    g1 = 1.0
    g2 = g_theta
    theta = best_theta

    opt_count = 0
    for i in range(iterations):
        u = torch.randn(theta.size()).type(torch.FloatTensor)
        u = u/torch.norm(u)
        ttt = theta+beta * u
        ttt = ttt/torch.norm(ttt)
        g1, count = fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2)
        opt_count += count
        if (i+1)%50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))
        gradient = (g1-g2)/torch.norm(ttt-theta) * u
        theta.sub_(alpha*gradient)
        theta = theta/torch.norm(theta)
        g2, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd = g2)
        opt_count += count

    #g2, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd = g2)
    distortion = torch.norm(g2*theta)
    target = model.predict(x0 + g2*theta)
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (distortion, target, query_count + opt_count, timeend-timestart))
    return x0 + g2*theta

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


def attack_mnist_single(model, train_dataset,image, label, target = None):
    show_image(image.numpy())
    print("Original label: ", label)
    print("Predicted label: ", model.predict(image))
    if target == None:
        adversarial = attack_untargeted(model, train_dataset, image, label, alpha = alpha, beta = beta, iterations = 5000)
    else:
        print("Targeted attack: %d" % target)
        adversarial = attack_targeted(model, train_dataset, image, label, target, alpha = alpha, beta = beta, iterations = 5000)
    show_image(adversarial.numpy())
    print("Predicted label for adversarial example: ", model.predict(adversarial))
    return torch.norm(adversarial - image)

def attack_mnist():
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    net = MNIST()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        
    #load_model(net, 'models/mnist_gpu.pt')
    load_model(net, 'models/mnist_cpu.pt')
    net.eval()

    model = net.module if torch.cuda.is_available() else net

    num_images = 10

    print("\n\n\n\n\n Running on {} random images \n\n\n".format(num_images))
    distortion_random_sample = 0.0

    for _ in range(num_images):
        idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        targets = list(range(10))
        targets.pop(label)
        target = random.choice(targets)
        #target = None   --> uncomment of untarget
        distortion_random_sample += attack_mnist_single(model, train_dataset, image, label, target)
    
    print("Average distortion on random {} images is {}".format(num_images, distortion_random_sample/num_images))

def attack_cifar10_single(model, train_dataset,image, label, target = None):
    print("Original label: ", label)
    print("Predicted label: ", model.predict(image))
    if target == None:
        adversarial = attack_untargeted(model, train_dataset, image, label, alpha = alpha, beta = beta, iterations = 5000)
    else:
        print("Targeted attack: %d" % target)
        adversarial = attack_targeted(model, train_dataset, image, label, target, alpha = alpha, beta = beta, iterations = 5000)
    print("Predicted label for adversarial example: ", model.predict(adversarial))
    return torch.norm(adversarial - image)

def attack_cifar10():
    train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()
    net = CIFAR10()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        
    #load_model(net, 'models/cifar10_gpu.pt')
    load_model(net, 'models/cifar10_cpu.pt')
    net.eval()

    model = net.module if torch.cuda.is_available() else net

    num_images = 10

    print("\n\n\n\n\n Running on {} random images \n\n\n".format(num_images))
    distortion_random_sample = 0.0

    for _ in range(num_images):
        idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        targets = list(range(10))
        targets.pop(label)
        target = random.choice(targets)
        #target = None   --> uncomment of untarget
        distortion_random_sample += attack_cifar10_single(model, train_dataset, image, label, target)
    
    print("Average distortion on random {} images is {}".format(num_images, distortion_random_sample/num_images))

if __name__ == '__main__':
    timestart = time.time()
    random.seed(0)
    attack_mnist()
    #attack_cifar10()
    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
