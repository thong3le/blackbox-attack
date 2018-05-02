import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from models import IMAGENET, MNIST, CIFAR10, load_imagenet_data, load_mnist_data, load_cifar10_data, load_model, show_image


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

    num_samples = 1000
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

        if beta <= 1e-6:
            break

        if (i+1)%10 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d alpha %.5f beta %.5f" % (i+1, g1, g2, g2, opt_count, alpha, beta))
            
        gradient = (g1-g2)/torch.norm(ttt-theta) * u
        
        # make sure the step-size is good
        # new_alpha = alpha
        # while new_alpha > 1e-6:
        #     new_theta = theta - new_alpha*gradient
        #     new_theta = new_theta/torch.norm(new_theta)
        #     new_g2, count = fine_grained_binary_search_targeted(model, x0, y0, target, new_theta, initial_lbd = g2)
        #     opt_count += count
        #     if new_g2 != float('inf'):
        #         g2 = new_g2
        #         theta = new_theta
        #         break
        #     new_alpha *= 0.8

        new_theta = theta - alpha * gradient
        new_theta = new_theta/torch.norm(new_theta)
        new_g2, count = fine_grained_binary_search_targeted(model, x0, y0, target, new_theta, initial_lbd = g2)
        opt_count += count

        if new_g2 > min(g2, g1):
            if g2 > g1:
                theta = ttt
                g2 = g1
        else:
            theta = new_theta
            g2 = new_g2

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

    num_samples = 10
    best_theta, g_theta = None, float('inf')
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
            initial_lbd = torch.norm(theta)
            theta = theta/torch.norm(theta)
            lbd, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print("--------> Found distortion %.4f" % g_theta)

    timeend = time.time()
    print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
  
    timestart = time.time()
    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta

    opt_count = 0
    q = 20
    for i in range(iterations):
        gradient = torch.zeros(theta.size())
        #if i>150:
        #    q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = torch.randn(theta.size()).type(torch.FloatTensor)
            u = u/torch.norm(u)
            ttt = theta+beta * u
            ttt = ttt/torch.norm(ttt)
            g1, count = fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0/q * gradient

        if (i+1)%50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))

        min_theta = theta
        min_g2 = g2
        new_alpha = alpha

        for t in range(10):
            new_theta = theta - new_alpha * gradient
            new_theta = new_theta/torch.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2)
            opt_count += count
            new_alpha = new_alpha * 1.1
            if new_g2 < min_g2:
                min_theta = new_theta 
                min_g2 = new_g2
            else:
                #print("Increase ", t)
                break

        if min_g2 >= g2:
            new_alpha = alpha
            for t in range(10):
                new_alpha = new_alpha * 0.9
                new_theta = theta - new_alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2)
                opt_count += count
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    #print("Decrease ", t)
                    break
            #print(i, min_g2, g2)
        alpha = new_alpha
        #print(alpha)
        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = theta.clone(), g2
        
        if alpha < 1e-4:
            break

    target = model.predict(x0 + g_theta*best_theta)
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
    return x0 + g_theta*best_theta

def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd
   
    if model.predict(x0+lbd*theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.05
        nquery += 1
        while model.predict(x0+lbd_hi*theta) == y0:
            lbd_hi = lbd_hi*1.05
            nquery += 1
            if lbd_hi > 100:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.95
        nquery += 1
        while model.predict(x0+lbd_lo*theta) != y0 :
            lbd_lo = lbd_lo*0.95
            nquery += 1

    while (lbd_hi - lbd_lo) > 1e-5:
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

def attack_mnist(alpha=0.2, beta=0.001, isTarget= False, num_attacks= 100):
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()

    dataset = train_dataset

    net = MNIST()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        
    load_model(net, 'models/mnist_gpu.pt')
    #load_model(net, 'models/mnist_cpu.pt')
    net.eval()

    model = net.module if torch.cuda.is_available() else net

    def single_attack(image, label, target = None):
        show_image(image.numpy())
        print("Original label: ", label)
        print("Predicted label: ", model.predict(image))
        if target == None:
            adversarial = attack_untargeted(model, dataset, image, label, alpha = alpha, beta = beta, iterations = 1000)
        else:
            print("Targeted attack: %d" % target)
            adversarial = attack_targeted(model, dataset, image, label, target, alpha = alpha, beta = beta, iterations = 1000)
        show_image(adversarial.numpy())
        print("Predicted label for adversarial example: ", model.predict(adversarial))
        return torch.norm(adversarial - image)

    print("\n\n Running {} attack on {} random  MNIST test images for alpha= {} beta= {}\n\n".format("targetted" if isTarget else "untargetted", num_attacks, alpha, beta))
    total_distortion = 0.0

    for _ in range(num_attacks):
        idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        target = None if not isTarget else random.choice(list(range(label)) + list(range(label+1, 10)))
        total_distortion += single_attack(image, label, target)
    print("Average distortion on random {} images is {}".format(num_attacks, total_distortion/num_attacks))


def attack_cifar10(alpha= 0.2, beta= 0.001, isTarget= False, num_attacks= 100):
    train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()
    dataset = train_dataset

    net = CIFAR10()
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        
    load_model(net, 'models/cifar10_gpu.pt')
    #load_model(net, 'models/cifar10_cpu.pt')
    net.eval()

    model = net.module if torch.cuda.is_available() else net

    def single_attack(image, label, target = None):
        print("Original label: ", label)
        print("Predicted label: ", model.predict(image))
        if target == None:
            adversarial = attack_untargeted(model, dataset, image, label, alpha = alpha, beta = beta, iterations = 1500)
        else:
            print("Targeted attack: %d" % target)
            adversarial = attack_targeted(model, dataset, image, label, target, alpha = alpha, beta = beta, iterations = 1500)
        print("Predicted label for adversarial example: ", model.predict(adversarial))
        return torch.norm(adversarial - image)

    print("\n\nRunning {} attack on {} random CIFAR10 test images for alpha= {} beta= {}\n\n".format("targetted" if isTarget else "untargetted", num_attacks, alpha, beta))
    total_distortion = 0.0

    samples =  [6411, 4360, 7753, 7413, 684, 3343, 6785, 7079, 2263]
    #for _ in range(num_attacks):
    for idx in samples:
        #idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        target = None if not isTarget else random.choice(list(range(label)) + list(range(label+1, 10)))
        total_distortion += single_attack(image, label, target)
    
    print("Average distortion on random {} images is {}".format(num_attacks, total_distortion/num_attacks))

def attack_imagenet(arch='resnet50', alpha=0.2, beta= 0.001, isTarget=False, num_attacks = 100):
    train_loader, test_loader, train_dataset, test_dataset = load_imagenet_data()
    dataset = test_dataset

    model = IMAGENET(arch)

    def attack_single(image, label, target = None):
        print("Original label: ", label)
        print("Predicted label: ", model.predict(image))
        if target == None:
            adversarial = attack_untargeted(model, dataset, image, label, alpha = alpha, beta = beta, iterations = 500)
        else:
            print("Targeted attack: %d" % target)
            adversarial = attack_targeted(model, dataset, image, label, target, alpha = alpha, beta = beta, iterations = 500)
        print("Predicted label for adversarial example: ", model.predict(adversarial))
        return torch.norm(adversarial - image)

    print("\nRunning {} attack on {} random IMAGENET test images for alpha= {} beta= {} using {}\n".format("targetted" if isTarget else "untargetted", num_attacks, alpha, beta, arch))
    total_distortion = 0.0

    for _ in range(num_attacks):
        idx = random.randint(100, len(test_dataset)-1)
        image, label = test_dataset[idx]
        print("\n\n======== Image %d =========" % idx)
        target = None if not isTarget else random.choice(list(range(label)) + list(range(label+1, 1000)))
        total_distortion += attack_single(image, label, target)
    
    print("Average distortion on random {} images is {}".format(num_attacks, total_distortion/num_attacks))

if __name__ == '__main__':
    timestart = time.time()
    random.seed(0)
    
    #attack_mnist(alpha=2, beta=0.001, isTarget= False, num_attacks= 10)
    attack_cifar10(alpha=5, beta=0.005, isTarget= False, num_attacks= 10)
    #attack_imagenet(arch='resnet50', alpha=0.05, beta=0.001, isTarget= False, num_attacks= 10)
    #attack_imagenet(arch='vgg19', alpha=0.05, beta=0.001, isTarget= False, num_attacks= 10)

    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
