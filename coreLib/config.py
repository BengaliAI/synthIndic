# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''

class augment_conf:
    rotation_angle_max    = 5
    rotation_exec_weights = [0.5,0.5]   
    warping_len_max_perc  = 20
    warping_exec_weights  = [0.4,0.6]
    curve_angle_min       = 30
    curve_angle_max       = 180
    curve_exec_weights    = [0.3,0.7]
    mask_negation_prec    = 30
    mask_negation_weights = [0.4,0.6]

class text_conf:
    lens                  = [1,2,3,4,5,6,7,8,9,10]
    weights               = [0.05,0.05,0.1,0.15,0.15,0.15,0.15,0.1,0.05,0.05]
    include_space         = True
    max_space             = 3
    space_weights         = [0.8,0.2]
    max_font_text_dim     = 256
    min_font_text_dim     = 20
    ext_weights           = [0.7,0.3]
    max_pad_perc          = 10
    min_pad_perc          = 1
    comp_weights          = [0.7,0.2,0.1]
    