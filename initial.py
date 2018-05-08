import time, sys
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from models import MNIST, CIFAR10, IMAGENET, SimpleMNIST, load_mnist_data, load_cifar10_data, load_imagenet_data, load_model, show_image

alpha = 0.2
#beta = 0.005
beta = 0.05
def count_neg_ratio(num_array):
    #print(num_array)
    count =0
    for i in num_array:
        if i<0:
            count+=1
    return float(count)/float(len(num_array))

def blended_attack(model,x0,y0):
    for _ in range(10):
        random_noise = torch.randn(x0.size())
        epsilons = np.linspace(0, 1, num=1000 + 1)[1:]
        for epsilon in epsilons:
            perturbed = (1 - epsilon) * x0 + epsilon * random_noise
            if model.predict(perturbed) != y0:
                return perturbed

def attack_targeted(model, train_loader, x0, y0, target, alpha = 0.1, beta = 0.001, iterations = 1000, batch_size = 10):
    """ Attack the original image and return adversarial example of target t
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
        t: target
    """ 
    array_beta = []
    o_alpha = 0.1
    beta = 0.01
    if (model.predict(x0) != y0):
        print("Fail to classify the image. No need to attack.")
        return x0
    # STEP I: find initial direction (theta, g_theta)
     
    image, label = Variable(x0.cuda()), y0 
    #print(predicted!=y0).nonzero()
    '''
    num_samples = 1000 
    best_theta, g_theta = None, float('inf')
    query_count = 0

    #print("Searching for the initial direction on %d samples: " % (num_samples))
    timestart = time.time()
    #samples = set(random.sample(range(len(train_dataset)), num_samples))
    '''
    timestart = time.time()
    b_best_lbd = float('inf')
    dim1 = x0.size()[0]
    dim3 = x0.size()[2]
    #for index in range(batch_size):
    for i, (xi, yi) in enumerate(train_loader):
        if i == 1:
            break
        xi,yi=xi.cuda(),yi.cuda()
        #temp_x0, temp_y0 = x0[index], y0[index]
        temp_x0 = x0 
        #temp_x0 = temp_x0.expand(100,1,28,28)
            
        #b_target = torch.LongTensor([target[index]]).expand(100).cuda()
        #b_target = torch.LongTensor([target]).expand(b_train_size).cuda()
        b_index = ( target == yi).nonzero().squeeze()
        if len(b_index.size()) == 0:
            continue
        xi = xi[b_index]
        #b_target = b_target[b_index]
        temp_x0 = temp_x0.expand(xi.size()).cuda()
        theta = xi - temp_x0
        initial_lbd = torch.norm(torch.norm(torch.norm(theta,2,1),2,1),2,1)
        initial_lbd = initial_lbd.unsqueeze(1).unsqueeze(2).expand(xi.size()[0],dim1,dim3).unsqueeze(3).expand(xi.size()[0],dim1,dim3,dim3)
        theta /= initial_lbd
        lbd, query_count = initial_fine_grained_binary_search_targeted(model, temp_x0, target, theta, initial_lbd)
        #print(lbd)    
        best_lbd, best_index = torch.min(lbd,0)
        #print(best_lbd)
        best_theta = theta[best_index]
        if best_lbd[0] < b_best_lbd:
            
            #print(model.predict(x0.cuda()+best_lbd*best_theta))
            b_best_lbd = best_lbd[0]
            b_best_theta = best_theta.clone()
            print("--------> Found g() %.4f" %b_best_lbd)

    best_theta, g_theta = b_best_theta.cpu(), b_best_lbd
     
    #print(model.predict(x0+g_theta*best_theta))
    timeend = time.time()
    print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (b_best_lbd, timeend-timestart, query_count))
    

    # STEP II: seach for optimal
    timestart = time.time()
    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta
    print(model.predict(x0+theta*g2))
    opt_count = 0
    torch.manual_seed(0)
    inner_size = 1
    for i in range(iterations):
        gradient = torch.zeros(theta.size())
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = torch.randn(theta.size())
            u = u/torch.norm(u)
            ttt = theta+beta * u
            ttt = ttt/torch.norm(ttt)
            g1, count = fine_grained_binary_search_local_targeted(model, x0, target, ttt, initial_lbd = g2, tol=beta/500)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0 /q*gradient
        temp_output = model.predict(x0+g2*theta)
        if (i+1)%100 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d alpha %.5f beta %.5f output %d" % (i+1, g1, g2, g2, opt_count, alpha, beta, temp_output))
        min_theta = theta
        min_g2 = g2
        new_alpha = alpha
        
        for t in range(15):
            new_theta = theta - new_alpha * gradient
            new_theta = new_theta/torch.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local_targeted(model, x0, target, new_theta, initial_lbd = min_g2, tol=beta/500)
            opt_count += count
            new_alpha *= 1.2
            if new_g2 < min_g2:
                min_theta = new_theta
                min_g2 = new_g2
            else:
                break

        if min_g2 >= g2:
            new_alpha = alpha
            for t in range(15):
                new_alpha = new_alpha * 0.8
                new_theta = theta - new_alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local_targeted(model, x0, target, new_theta, initial_lbd = min_g2, tol=beta/500) 
                #new_g2 += eps
                opt_count += count
                if new_g2 < g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    #print("Decrease ", t)
                    break
            if t==14:
                 print("line search reach boundary, newalpha = %lf"%new_alpha)
                 print("(g1, g2) = ", (g1, g2),  "min_g1 = ", min_g1)
        alpha = new_alpha
        #print(alpha)
        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1
            #print("warning: min_ttt is smaller")

        if g2 < g_theta:
            best_theta, g_theta = theta.clone(), g2
        #else:
            #print("warning: not moving, g2 %lf gtheta %lf"%(g2, g_theta))
        
        if alpha < 1e-4:
            alpha = 0.2
            print("warning: alpha is %lf"%alpha)
            beta = beta*0.5
            if (beta < 0.00005):
                break
            #break

    g2, count = fine_grained_binary_search_local_targeted(model, x0, target, theta, initial_lbd = g2)
    #distorch = torch.norm(g2*theta)
    out_target = model.predict(x0 + g2*theta)  # should be the target
    timeend = time.time()
    print("\nAdversarial Example Tageted %d Found Successfully: distortion %.4f target %d queries %d alpha %.5f beta %.5f \nTime: %.4f seconds" % (target, g2, out_target, query_count + opt_count, alpha, beta, timeend-timestart))
    return x0 + g2*theta, query_count + opt_count

def fine_grained_binary_search_local_targeted(model, x0, t, theta, initial_lbd = 1.0, tol=1e-5):
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

    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) == t:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def initial_fine_grained_binary_search_targeted(model, x0, target, theta, initial_lbd = 1.0):
    dim1 = x0.size()[1]
    dim3 = x0.size()[2]    
    nquery = 0
    initial_lbd = torch.ones(theta.size()).cuda()
    lbd = initial_lbd
    limit = torch.ones(lbd.size()).cuda()
    predicted = model.predict_batch(x0+ lbd * theta)
    #nquery += 10
    candidate = (predicted != target).nonzero().view(-1)
    while len(candidate.size())>0:
        lbd[candidate] = lbd[candidate].mul(1.05)
        limit.resize_(candidate.size())
        if torch.max(lbd) > 100: 
            break
        predicted = model.predict_batch(x0+ lbd * theta)
        nquery += candidate.size()[0]
        candidate = (predicted != target).nonzero().view(-1)
   
    #lbd = torch.clamp(lbd,0,100)
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
        candidate = (predicted == target).nonzero().view(-1)
        nquery += num_intervals-1 - candidate.size()[0]
        if len(candidate.size())==0:
            lbd_hi_index[i]=0
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
        
        candidate_y = (predicted == target).nonzero().view(-1)
        candidate_n = (predicted != target).nonzero().view(-1)
        if len(candidate_y.size())>0:
            lbd_hi[candidate_y] = lbd_mid[candidate_y]
        if len(candidate_n.size())>0:
            lbd_lo[candidate_n] = lbd_mid[candidate_n]
    return lbd_hi, nquery

def fine_grained_binary_search_targeted(model, x0, target, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd
    dim1 = x0.size()[0]
    dim3 = x0.size()[2] 
 
    while model.predict(x0+lbd*theta) != target:
        lbd *= 1.05
        nquery +=1

    if lbd> 100:
        return float('inf')
    
    num_intervals = 100
    
    lbd_hi = lbd
    lbd_hi_index = 0
    lambdas = torch.linspace(0.0, lbd, num_intervals)[1:]
    lambdas[98] = lbd
    #lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
    #lambdas = torch.from_numpy(lambdas).type(torch.FloatTensor)
    #print(lambdas[98], lbd)
    #print(model.predict(x0+lambdas[98]*theta))
    temp_lbd = lambdas.unsqueeze(1).unsqueeze(2).expand(num_intervals-1,dim1,dim3).unsqueeze(3).expand(num_intervals-1,dim1,dim3,dim3)
    #print(temp_lbd[98][0][0][0])
    temp_theta = theta.expand(num_intervals-1,dim1,dim3,dim3)
    temp_x0 = x0.expand(num_intervals-1,dim1,dim3,dim3)
    predicted = model.predict_batch(temp_x0+temp_lbd*temp_theta)
    #print(predicted[98])
    candidate = (predicted == target).nonzero().view(-1)
    if len(candidate.size())==0:
        lbd_hi_index = 0
        lbd_hi = lbd
        print("aa")
    else:
        lbd_hi_index = torch.min(candidate)
        lbd_hi = lambdas[lbd_hi_index]
        nquery += num_intervals-1 - candidate.size()[0]
    lbd_lo = lambdas[lbd_hi_index - 1]
    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0+lbd_mid*theta) == target:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery



def attack_untargeted(model, train_loader, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000):
    """ Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
    """
    eps = 1e-6
    o_alpha = alpha
    array_alpha = []
    array_beta = []
    if (model.predict(x0) != y0):
        print("Fail to classify the image. No need to attack.")
        return x0, 1

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
        '''
        xi,yi=xi.cuda(),yi.cuda()
        temp_x0, temp_y0 = x0, y0
        predicted = model.predict_batch(xi)
        b_index = (predicted !=y0).nonzero().squeeze()
        if len(b_index.size()) == 0:
            continue
        xi = xi[b_index]
        yi = yi[b_index]
        temp_x0 = temp_x0.expand(xi.size()).cuda()
        theta = xi - temp_x0
        '''
        theta = torch.randn(50,3,32,32).cuda() 
        temp_x0, temp_y0 = x0, y0
        temp_x0 = temp_x0.expand(theta.size()).cuda()
        initial_lbd = torch.norm(torch.norm(torch.norm(theta,2,1),2,1),2,1)
        initial_lbd = initial_lbd.unsqueeze(1).unsqueeze(2).expand(theta.size()[0],dim1,dim3).unsqueeze(3).expand(theta.size()[0],dim1,dim3,dim3)
    
        #print(initial_lbd)
        theta /= initial_lbd
        lbd, query_count = initial_fine_grained_binary_search(model, temp_x0, y0, theta)
        #print(lbd,xi,yi,theta)
        best_lbd, best_index = torch.min(lbd,0)
        #best_index = 8
        best_lbd = lbd[best_index]
        best_theta = theta[best_index]
        #print(model.predict(best_x),best_y)
        '''
        if best_lbd[0] < b_best_lbd:
            b_best_lbd = best_lbd[0]
            b_best_theta = best_theta.clone()
            print("--------> Found g() %.4f" %b_best_lbd)
        '''
    #best_theta, g_theta = best_theta.cpu(), best_lbd[0]+eps
    best_theta, g_theta = best_theta.cpu(), (best_lbd+eps)[0]
    '''
    new_adv = torch.clamp(x0+best_theta*g_theta,0,1)
    new_adv = Variable(new_adv,volatile=True).view(1,3,32,32).cuda()
    print(model(new_adv))
    '''
    timeend = time.time()
    print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
    
    # STEP II: seach for optimal
    timestart = time.time()
    '''
    #initial = blended_attack(model,x0,y0)
    initiial = 
    theta = initial - x0    
    g2 = torch.norm(theta)
    theta /= g2
    '''
    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta
    #print(theta,g2)
    print(model.predict(x0+theta*g2))
    opt_count = 0
    o_g2 = 0
    torch.manual_seed(0)
    inner_size = 1
    per_step = 100
    for i in range(iterations):
        if (i+1)<=1000:
            gradient = torch.zeros(theta.size())
            q = 10 
            min_g1 = float('inf')
            for _ in range(q):
                u = torch.randn(theta.size()).type(torch.FloatTensor)
                u = u/torch.norm(u)
                ttt = theta+beta * u
                ttt = ttt/torch.norm(ttt)
                #u = ttt-theta
                #beta = torch.norm(u)
                g1, count = fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2, tol=beta/500) 
                #g1 += eps
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient

            if (i+1)%100 == 0:
                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d output %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count, model.predict(x0+g2*theta)))

            min_theta = theta
            min_g2 = g2
            new_alpha = alpha

            for t in range(15):
                new_theta = theta - new_alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500) 
                #new_g2 += eps
                opt_count += count
                new_alpha = new_alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    #print("Increase ", t)
                    break

            if min_g2 >= g2:
                new_alpha = alpha
                for t in range(15):
                    new_alpha = new_alpha * 0.25
                    new_theta = theta - new_alpha * gradient
                    new_theta = new_theta/torch.norm(new_theta)
                    new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500) 
                    #new_g2 += eps
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        #print("Decrease ", t)
                        break
                if t==14:
                     print("line search reach boundary, newalpha = %lf"%new_alpha)
                     print("(g1, g2) = ", (g1, g2),  "min_g1 = ", min_g1)
            alpha = new_alpha
            #print(alpha)
            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1
                print("warning: min_ttt is smaller")

            if g2 < g_theta:
                best_theta, g_theta = theta.clone(), g2
            else:
                print("warning: not moving, g2 %lf gtheta %lf"%(g2, g_theta))
            
            if alpha < 1e-4:
                alpha = 1.0
                print("warning: alpha is %lf"%alpha)
                beta = beta*0.1
                if (beta < 0.0005):
                    break
                #break

        else:
            if (i+1) == 1001:
                break
                print(alpha)
                alpha = o_alpha
                #alpha /= q
            if (i+2)%100==0:
                ''' 
                beta_ratio = count_neg_ratio(array_beta)
                #print(beta_ratio)
                if beta_ratio<0.25:
                    beta *= 0.8
                elif beta_ratio>0.4:
                    beta *= 1.2
                    #alpha *= 1.2
                '''
                #print(array_alpha)
                alpha_ratio = count_neg_ratio(array_alpha)
                beta_ratio = count_neg_ratio(array_beta)
                print(alpha_ratio, beta_ratio)
                if beta_ratio==0:
                    beta /= 2
                if alpha_ratio == 0:
                    #inner_size = 10 
                    break 
                if alpha_ratio<0.1:
                    alpha *= 0.8
                    #beta == 0.001
                elif alpha_ratio>0.8:
                    alpha *= 1.25
                
                array_alpha= []
                array_beta = []
                
            #if (i+1)==3000:
            #    beta = 0.01
            avg_grad = torch.zeros(theta.size())
            for _ in range(inner_size): 
                u = torch.randn(theta.size())
                u = u/torch.norm(u)
                g2, count = fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = g2)
                opt_count += count
                ttt = theta+beta * u
                ttt = ttt/torch.norm(ttt)
                g1, count = fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2)
                opt_count += count
                temp_output = model.predict(x0+g2*theta)
                array_beta.append(g1-g2) 
                #gradient = (g1-g2)/torch.norm(ttt-theta) * u
                gradient = (g1-g2)/beta * u
                avg_grad += gradient
            if (i+1)%per_step == 0:
                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d alpha %.5f beta %.5f output %d" % (i+1, g1, g2, g2, opt_count, alpha, beta, temp_output))
            gradient = avg_grad/inner_size
            temp_theta = theta - alpha*gradient
            temp_theta /= torch.norm(temp_theta)
            g3, count = fine_grained_binary_search_local(model, x0, y0, temp_theta, initial_lbd = g2)
            array_alpha.append(g3-g2)
            if g3 > g1:
                #print("aa")
                theta = ttt
                #print(fine_grained_binary_search_targeted(model, x0, y0, ttt, initial_lbd = g2))
            else:
                theta.sub_(alpha*gradient)
                theta /= torch.norm(theta)
        o_g2 = g2
   
    g2, count = fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = g2)
    out_target = model.predict(x0 + g2*theta)  # should be the target
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d alpha %.5f beta %.5f \nTime: %.4f seconds" % (g2, out_target, query_count + opt_count, alpha, beta, timeend-timestart))
    return x0 + g2*theta, query_count + opt_count

def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = 1.0, tol = 1e-5):
    nquery = 0
    lbd = initial_lbd
    if model.predict(x0+lbd*theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while model.predict(x0+lbd_hi*theta) == y0:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi >200:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while model.predict(x0+lbd_lo*theta) != y0 :
            lbd_lo = lbd_lo*0.99
            nquery += 1
    while (lbd_hi - lbd_lo) > tol:
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
    #nquery += 100
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
        if len(candidate.size())==0:
            lbd_hi_index[i] = 0
            #print(lbd[i][0],lbd_hi[i])
            lbd_hi[i] = lbd[i][0][0][0]
            #return float('inf'), nquery
        else:
            lbd_hi_index[i] = torch.min(candidate)
            lbd_hi[i] = lambdas[i][lbd_hi_index[i]]
            nquery += num_intervals-1 - candidate.size()[0]
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
    dim1 = x0.size()[0]
    dim3 = x0.size()[2]
    while model.predict(x0+lbd*theta) == y0:
        lbd *= 1.05
        nquery +=1

    if lbd> 1000:
        return float('inf')
    
    num_intervals = 100
    
    lbd_hi = lbd
    lbd_hi_index = 0
    lambdas = torch.linspace(0.0, lbd, num_intervals)[1:]
    lambdas[98] = lbd
    #lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
    #lambdas = torch.from_numpy(lambdas).type(torch.FloatTensor)
    #print(lambdas[98], lbd)
    #print(model.predict(x0+lambdas[98]*theta))
    temp_lbd = lambdas.unsqueeze(1).unsqueeze(2).expand(num_intervals-1,dim1,dim3).unsqueeze(3).expand(num_intervals-1,dim1,dim3,dim3)
    #print(temp_lbd[98][0][0][0])
    temp_theta = theta.expand(num_intervals-1,dim1,dim3,dim3)
    temp_x0 = x0.expand(num_intervals-1,dim1,dim3,dim3)
    predicted = model.predict_batch(temp_x0+temp_lbd*temp_theta)
    #print(predicted[98])
    candidate = (predicted!=y0).nonzero().view(-1)
    if len(candidate.size())==0:
        lbd_hi_index = 0
        lbd_hi = lbd
        print("aa")
    else:
        lbd_hi_index = torch.min(candidate)
        lbd_hi = lambdas[lbd_hi_index]
        nquery += num_intervals-1 - candidate.size()[0]
    lbd_lo = lambdas[lbd_hi_index - 1]
    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if model.predict(x0+lbd_mid*theta) !=y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery


def attack_single(model, train_loader, image, label, target = None, alpha=0.2, beta=beta):
    #show_image(image.numpy())
    print("Original label: ", label)
    print("Predicted label: ", model.predict(image))
    if target == None:
        adversarial, count = attack_untargeted(model, train_loader, image, label, alpha = alpha, beta = beta, iterations = 20000)
    else:
        print("Targeted attack: %d" % target)
        adversarial, count = attack_targeted(model, train_loader, image, label, target, alpha = alpha, beta = beta, iterations = 5000)
    #show_image(adversarial.numpy())
    print("Predicted label for adversarial example: ", model.predict(adversarial))
    return torch.norm(adversarial - image), count

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

    distortion_random_sample = 0.0
    idx_a = [6312, 6891, 664, 4243, 8377, 7962, 6635, 4970, 7809, 5867, 9559, 3579, 8269, 2282, 4618, 2290, 1554, 4105, 8726, 9862, 2408, 5082, 1619, 1209, 5410, 7736, 9172, 1650, 5797, 7114, 5181, 3351, 9053, 7816, 7254, 8542, 4268, 1021, 8990, 231, 1529, 6535, 19, 8087, 5459, 3997, 5329, 1032, 3131, 9299, 3633, 3910, 2335, 8897, 7340, 1495, 1319, 5244, 8323, 8017, 1787, 4939, 9032, 4770, 2045, 8970, 5452, 8853, 3330, 9883, 8966, 9628, 4713, 7291, 1502, 9770, 6307, 5195, 9432, 3967, 4757, 3013, 3103, 3060, 541, 4261, 7808, 1132, 1472, 2134, 2451, 634, 1315, 8858, 6411, 8595, 4516, 8550, 3859, 3526]   
    #idx_a=[2450, 633, 1314, 8857, 6410, 8594, 4515, 8549, 3858, 3525, 6411, 4360, 7753, 7413, 684, 3343, 6785, 7079, 2263]
    #idx_a = [2289, 5409, 3350, 1786, 7807, 2450, 4515, 3858]
    #idx_a = [6411, 4360, 7753, 7413, 684, 3343, 6785, 7079, 2263]
    #idx_a = [7808]
    num_images = len(idx_a)
    print("\n\n\n\n\n Running on {} random images \n\n\n".format(num_images))
    sum_queries = 0
    sum_dis = 0.0
    #for _ in range(num_images):
    for idx in idx_a:
        #idx = random.randint(100, len(test_dataset)-1)
        #idx = 5474

        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        target = (label+1)%10
        #target = 3
        #target = None   #--> uncomment of untarget
        distortion_random_sample, count= attack_single(model, train_loader, image, label, target, alpha)
        sum_dis += distortion_random_sample
        sum_queries += count
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

    
    distortion_random_sample = 0.0
    idx_a =  [6311, 6890, 663, 4242, 8376, 7961, 6634, 4969, 7808, 5866, 9558, 3578, 8268, 2281, 4617, 2289, 1553, 4104, 8725, 9861, 2407, 5081, 1618, 1208, 5409, 7735, 9171, 1649, 5796,7113, 5180, 3350, 9052, 7815, 7253, 8541, 4267, 1020, 8989, 230, 1528, 6534, 18, 8086, 5458, 3996, 5328, 1031, 3130, 9298, 3632, 3909, 2334, 8896, 7339, 1494, 1318, 5243, 8322, 8016, 1786, 4938, 9031, 4769, 2044, 8969, 5451, 8852, 3329, 9882, 8965, 9627, 4712, 7290, 1501, 9769, 6306, 5194, 9431, 3966, 4756, 3012, 3102, 3059, 540, 4260, 7807, 1131, 1471, 2133]
    #idx_a=[2450, 633, 1314, 8857, 6410, 8594, 4515, 8549, 3858, 3525, 6411, 4360, 7753, 7413, 684, 3343, 6785, 7079, 2263]
    #idx_a = [2289, 5409, 3350, 1786, 7807, 2450, 4515, 3858]
    #idx_a = [6411, 4360, 7753, 7413, 684, 3343, 6785, 7079, 2263]
    #idx_a = [7808]
    num_images = len(idx_a)
    print("\n\n\n\n\n Running on {} random images \n\n\n".format(num_images))
    sum_queries = 0
    sum_dis = 0.0
    #for _ in range(num_images):
    for idx in idx_a:
        #idx = random.randint(100, len(test_dataset)-1)
        #idx = 5474

        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        target = (label+1)%10
        #target = 3
        #target = None   #--> uncomment of untarget
        distortion_random_sample, count= attack_single(model, train_loader, image, label, target, alpha)
        sum_dis += distortion_random_sample
        sum_queries += count
    #print("\n\n\n\n\n Running on first {} images \n\n\n".format(num_images))
    print("Average distortion on random {} images is {} using {} queries".format(num_images, sum_dis/num_images, sum_queries/num_images))
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
def attack_imagenet(arch='resnet50', alpha=0.2, beta= 0.001, isTarget=False, num_attacks = 100):
    train_loader, test_loader, train_dataset, test_dataset = load_imagenet_data()
    dataset = test_dataset

    model = IMAGENET(arch)

    distortion_random_sample = 0.0
    num_images = 1
    idx_a= [25248, 49674]
    for idx in idx_a:
        #idx = random.randint(100, len(test_dataset)-1)
        #idx = 3743
        image, label = test_dataset[idx]
        print("\n\n\n\n======== Image %d =========" % idx)
        targets = list(range(1000))
        targets.pop(label)
        #target = random.choice(targets)
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
    #attack_imagenet()
    timeend = time.time()
    print("\n\nTotal running time: %.4f seconds\n" % (timeend - timestart))
