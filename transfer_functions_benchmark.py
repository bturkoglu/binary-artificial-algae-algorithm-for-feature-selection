


#import skfuzzy
import numpy as np
from math import pi
from scipy.special import erf
import matplotlib.pylab as plt


##______________________S-shaped transfer functions_______________________

def s1(x):
    
    s1=1 / (1 + np.exp(-2*x))
    
    return s1

def s2(x):
    s2 = 1 / (1 + np.exp(-x))  
    return s2
# s2 is called logistic function and can be imported using scipy.special.expit(x) library

def s3(x):
    s3=1 / (1 + np.exp(-x/3))
    return s3


def s4(x):
    s4=1 / (1 + np.exp(-x/2))
    return s4







x = np.arange(-8, 8, 0.1) 
# x is used inside this script and will be replaced by a binary individual (1-d binary vector)
#when the transfer function is called or imported in the optimizers scripts


#____________
























  
  
    
    
