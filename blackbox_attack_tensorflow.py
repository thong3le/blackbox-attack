## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time

from setup_mnist import MNIST, MNISTModel


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


# alpha = 0.018   # learning_reate
 #   beta = 0.05    # smoothing parameter
    #b = 0.9           # scaling to balance bias and variance 

def attack(model, images, x0, y0, alpha = 0.01, beta = 0.02, iterations = 100):
    """ find the adversarial example for given image (x0, y0)
        model -> tensorflow model
        images -> set of images from train or test dataset
        x0 -> orginal image that we want to find adversarial example for, dimension 1x1x28x28
        y0 -> label of x0, dimension 1x1
    """
    
    if (model.predict(x0) != y0):
        print("Fail to classify the image. No need to attack.")
        return

    theta = initial_direction(model, images, x0, y0)

    timestart = time.time()
    query_count = 0
    g1 = g2 = 1.0  # g2 = g(theta + beta*u), g1 = g(theta)
    
    for i in range(iterations):
        u = np.random.uniform(size=theta.shape)
        g1, count1 = backtracking_line_search(model, x0, y0, theta + beta * u, initial_lbd = g1)
        g2, count2 = backtracking_line_search(model, x0, y0, theta, initial_lbd = g2)
        gradient = (g1-g2)/beta * u
        if (i+1)%10 == 0:
            target =  model.predict(x0 + g2 * theta)[0]
            #print(target, g1, g2, np.linalg.norm(g2*theta))
            print("Iteration %3d: g(theta + beta*u) %.4f g(theta) %.4f distance %.4f target %d" % (i+1, g1, g2, np.linalg.norm(g2*theta), target))
        theta -= alpha*gradient
        query_count += count1 + count2 + 1

    timeend = time.time()
    print("Number of queries: %d, time %.4f seconds" % (query_count, timeend-timestart))
    return x0 + g2 * theta

def initial_direction(model, images, x0, y0):
    print("Find the initial direction: ")
    timestart = time.time()
    predict = model.predict(images)
    timeend = time.time()
    options = (images-x0)[predict!=y0]
    distances = np.linalg.norm(options.reshape(options.shape[0], -1), ord=2, axis=1, keepdims=True)
    idx = np.argmax(distances)
    theta = options[idx:idx+1]
    
    print("-----> Found the closest point outside of the region, distance %.4f in %.4f seconds" % (np.linalg.norm(theta), timeend-timestart))
    return theta

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

if __name__ == "__main__":
    with tf.Session() as sess:
        data, model =  MNIST(), MNISTModel("models/mnist_cw_tf", sess)
        train_data = data.train_data
        train_labels = data.train_labels
        test_data = data.test_data
        test_labels = data.test_labels
        
        idx = np.random.random_integers(len(train_data))

        image = train_data[idx:idx+1]
        label = np.argmax(train_labels[idx:idx+1], axis=1)

        

        adv = attack(model, train_data, image, label)

        print("Original Image with prediction ", model.predict(image)[0])
        show(image)
        print("Adversarial Image with prediction ", model.predict(adv)[0])
        show(adv)

