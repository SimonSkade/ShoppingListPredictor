import random
import numpy as np

def f(p_x, p_0, a_x, a_prev):
    if a_x == a_prev:
        if a_x == 1:
            return (p_x * p_0)**0.75
        else:
            return 1-((1-p_x)*(1-p_0))**0.75
    else:
        if a_x == 1:
            return p_0**1.5
        else:
            return 1-(1-p_0)**1.5


def generate(probs, prods):
    n = 28
    pxs = np.zeros((n,len(prods)))
    data = np.zeros((n,len(prods)))
    for i, prob in enumerate(probs):
        data[0][i] = np.random.choice([0,1], p=[1-prob,prob])
        px = f(prob,prob,data[0][1],-1)
        for j in range(1,n):
            data[j][i] = np.random.choice([0,1], p=[1-px,px])
            pxs[j][i] = px
            px = f(px,prob,data[j][i], data[j-1][i])
    return data, pxs

def generateFromGenFile(file):
    file = open(file, "r") 
    lines = file.readlines()
    lines = np.array([lines[i].replace(" ", "").strip().upper().split(",") for i in range(len(lines))])

    features = lines[:,0]

    num_features = len(features)

    vectors = []
    pxes = []

    for user in range(1,6):
        probs = lines[:,user]
        data, pxs = generate(probs.astype(np.float),features)
        pxes.append(pxs)
        vectors.append(data)
        
    return features, vectors, pxes