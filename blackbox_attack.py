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


def attack_targeted(model, train_dataset, x0, y0, t, alpha = 0.2, beta = 0.001):
    """ Attack the original image and return adversarial example of target t
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        t: target
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
        if model.predict(xi) == t:
            theta = xi - x0
            lbd, count = fine_grained_binary_search_target(model, x0, y0, t, theta)
            query_count += count
            distortion = torch.norm(lbd*theta)
            if distortion < best_distortion:
                best_theta, g_theta = theta, lbd
                best_distortion = distortion
                print("--------> Found distortion %.4f and g_theta = %.4f" % (best_distortion, g_theta))

    timeend = time.time()
    print("==========> Found best distortion %.4f and g_theta = %.4f in %.4f seconds using %d queries" % (best_distortion, g_theta, timeend-timestart, query_count))

    timestart = time.time()

    iterations = 5000
    g1 = 1.0
    g2 = g_theta
    theta = best_theta

    opt_count = 0
    for i in range(iterations):
        g2, count = fine_grained_binary_search_target(model, x0, y0, t, theta, initial_lbd = g2)
        opt_count += count
        
        u = torch.randn(theta.size()).type(torch.FloatTensor)
        u = u/torch.norm(u)
        ttt = theta+beta * u
        ttt = ttt/torch.norm(ttt)
        #opt_count += 1
        #print("opt_count: {}".format(opt_count))
        #if model.predict(x0 + g2*ttt) == t:
        #   break

        g1, count = fine_grained_binary_search_local_target(model, x0, y0, t, ttt, initial_lbd = g2)
        opt_count += count
        if (i+1)%1 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))
        gradient = (g1-g2)/torch.norm(ttt-theta) * u
        theta.sub_(alpha*gradient)
        theta = theta/torch.norm(theta)

    g2, count = fine_grained_binary_search_target(model, x0, y0, t, theta, initial_lbd = g2)
    distortion = torch.norm(g2*theta)
    target = model.predict(x0 + g2*theta)
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (distortion, target, query_count + opt_count, timeend-timestart))
    return x0 + g2*theta

def fine_grained_binary_search_local_target(model, x0, y0, t, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd
   
    if model.predict(x0+lbd*theta) != t:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while model.predict(x0+lbd_hi*theta) != t:
            lbd_hi = lbd_hi*1.01
            nquery += 1
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
        if model.predict(x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def fine_grained_binary_search_target(model, x0, y0, t, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd

    # lead to bug
    print(lbd)
    while model.predict(x0 + lbd*theta) != t:
        lbd *= 2.0
        print(lbd)
        nquery += 1

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

def attack_untargeted(model, train_dataset, x0, y0, alpha = 0.2, beta = 0.001):
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
            #query_count += query_search_each
            lbd, count = fine_grained_binary_search(model, x0, y0, theta)
            query_count += count
            distortion = torch.norm(lbd*theta)
            if distortion < best_distortion:
                best_theta, g_theta = theta, lbd
                best_distortion = distortion
                print("--------> Found distortion %.4f and g_theta = %.4f" % (best_distortion, g_theta))

    timeend = time.time()
    print("==========> Found best distortion %.4f and g_theta = %.4f in %.4f seconds using %d queries" % (best_distortion, g_theta, timeend-timestart, query_count))

    #query_limit -= query_count

  
    timestart = time.time()

    #query_search_each = 200  # limit for each lambda search
    #iterations = (query_limit - query_search_each)//(2*query_search_each)
    iterations = 5000
    g1 = 1.0
    g2 = g_theta
    theta = best_theta

    opt_count = 0
    for i in range(iterations):
        u = torch.randn(theta.size()).type(torch.FloatTensor)
        u = u/torch.norm(u)
        g2, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd = g2)
        opt_count += count
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


def attack_mnist():
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    net = MNIST()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        
    #load_model(net, 'models/mnist_gpu.pt')
    load_model(net, 'models/mnist.pt')
    net.eval()

    model = net.module if torch.cuda.is_available() else net

    num_images = 50

    print("\n\n\n\n\n Running on first {} images \n\n\n".format(num_images))

    distortion_fix_sample = 0.0

    for i, (image, label) in enumerate(test_dataset):
        if i >= num_images:
            break
        print("\n\n\n\n======== Image %d =========" % i)
        show_image(image.numpy())
        print("Original label: ", label)
        print("Predicted label: ", model.predict(image))
        adversarial = attack_untargeted(model, train_dataset, image, label, alpha = alpha, beta = beta)
        # target = 1
        #adversarial = attack_targeted(model, train_dataset, image, label, target, alpha = alpha, beta = beta)
        show_image(adversarial.numpy())
        print("Predicted label for adversarial example: ", model.predict(adversarial))
        distortion_fixsample += torch.norm(adversarial - image)

    print("\n\n\n\n\n Running on {} random images \n\n\n".format(num_images))

    distortion_random_sample = 0.0

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
        distortion_fixsample += torch.norm(adversarial - image)

    print("\n\nAverage distortion on first {} images is {}".format(num_images, distortion_fixsample/num_images))
    print("Average distortion on random {} images is {}".format(num_images, distortion_fixsample/num_images))


if __name__ == '__main__':
    timestart = time.time()
    attack_mnist()
    #attack_cifar10()
    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
