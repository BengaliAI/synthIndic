
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import numpy as np
import random
from .config import config
#--------------------
# paper ops
#--------------------
def mask_negation(mask):
    '''
        negate random pixels
    '''
    flat_mask=mask.flatten()
    idx=np.where(flat_mask>0)[0]
    neg_size=random.randint(10,config.mask_neg)/100
    idx=np.random.choice(idx, size=int(idx.size*neg_size), replace=False)
    flat_mask[idx]=0
    mask=flat_mask.reshape(mask.shape)
    return mask

def partial_negation(mask):
    '''
        negate clusterwise
    '''
    pass

def density_negation(mask):
    '''
        negate density
    '''
    pass

#--------------------
# scene ops
#--------------------
def add_boder(mask):
    '''
        adding border dialated: not needed in reality
    '''
    pass

def add_shadow(mask):
    '''
        adding border dialated: not needed in reality
    '''
    pass
