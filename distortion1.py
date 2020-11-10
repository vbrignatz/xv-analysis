import numpy as np 
import sys
import pandas
import time

from scipy.spatial.distance import cosine

def pd_reat_mat(filename):
    """ Lit les xv contenus dans un fichier et les retournes sous forme de matrice numpy"""
    mat = pandas.read_csv(filename, delimiter=" ", header=None)
    return mat.to_numpy()

def decision(xv1, xv2, e=0.73):
    """ Retourne True si les 2 xv partagent le meme locuteur, False sinon """
    return True if np.abs(cosine(xv1,xv2)) <= e else False

def distortion_1(X, Y, i, v=False):
    """ Cette distortion consiste a eloigner X[i] de Y[i] si X et Y on le meme locuteur,
    ou a raprocher X[i] de Y[i] si X et Y on un locuteur different"""
    d_init = decision(X, Y)
    for n in range(25):
        d = decision(X, Y)
        if v:
            print(f"X[i]={X[i]:.10f} Y[i]={Y[i]:.10f} cosine={cosine(X, Y):.10f} decision={d}")
        
        if d != d_init:
            return n

        if d_init:
            X[i] += (X[i] - Y[i])/2
        else:
            X[i] -= (X[i] - Y[i])/2
    return None

if __name__=="__main__":        
    if(len(sys.argv)!=2):
        print ("~~> Use : scoring.py <test_file>")
        sys.exit(0)

    testData=pd_reat_mat(sys.argv[1])

    data = pd_reat_mat(sys.argv[1])
    data = data / np.expand_dims(np.linalg.norm(data, axis=1), axis=1) # normalisation des donnees

    distortion_1(data[0].copy(), data[1].copy(), 0, v=True)

    for i in range(256):
        print(f"[{i}] : ", distortion_1(data[0].copy(), data[1].copy(), i))


