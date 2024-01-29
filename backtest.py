import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import pickle
import math
from statistics import NormalDist

def poisson_approx(k, lambda_):
    return (NormalDist(lambda_, math.sqrt(lambda_)).cdf(k+0.5) - NormalDist(lambda_, math.sqrt(lambda_)).cdf(k-0.5))

def poisson(k, lambda_):
    return (math.exp(-lambda_) * lambda_**k / math.factorial(k))

#thru 5OTs
def OT_dist(pace, h_rtg, a_rtg):
    ot_dist = []
    for i in range(25):
        ot_dist.append([])
        for j in range(25):
            ot_dist[i].append(poisson(i, h_rtg*pace/100/9.6) * poisson(j, a_rtg*pace/100/9.6))
    
    dist = []
    for i in range(70):
        dist.append([])
        for j in range(70):
            dist.append(0)
    
    return
    


#independent marginal poissons (normal approx)
#OT pace and rtg assumed to be same proportionally to reg time
def score_joint_pmf(pace, h_rtg, a_rtg):
    dist = []
    for i in range(225):
        dist.append([])
        for j in range(225):
            p = poisson_approx(i, h_rtg*pace/100) * poisson_approx(j, a_rtg*pace/100)
            if (i != j or p == 0):
                dist[i].append(p)
            else:
                pass
    
    return (dist)

