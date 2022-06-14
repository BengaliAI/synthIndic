# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import cv2 
import numpy as np 
import random
from PIL import Image
import math
from .utils import randColor
import blend_modes
from .config import config

BLENDS      =[blend_modes.normal,
              blend_modes.darken_only,
              blend_modes.multiply,
              blend_modes.hard_light,
              blend_modes.difference,
              blend_modes.grain_extract,
              blend_modes.soft_light,
              blend_modes.lighten_only,
              blend_modes.dodge,
              blend_modes.addition,
              blend_modes.overlay,
              blend_modes.grain_merge]
#----------------------------------------
# back grounds
#----------------------------------------
def get_scene_back(img,back_paths):
    '''
        creates scene background 
    '''
    hi,wi=img.shape
    back_path=random.choice(back_paths)
    back=cv2.imread(back_path)
    hb,wb,_=back.shape
    x=random.randint(0,wb-wi)
    y=random.randint(0,hb-hi)
    back=back[y:y+hi,x:x+wi]
    return back
    
def gaussian_noise(height, width):
    """
        Create a background with Gaussian noise (to mimic paper)
    """
    # We create an all white image
    image = np.ones((height, width)) * 255
    # We add gaussian noise
    cv2.randn(image, 235, 10)
    image=np.array(Image.fromarray(image).convert("RGB"))
    return image

def quasicrystal(height, width):
    """
        Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """

    image = Image.new("L", (width, height))
    pixels = image.load()

    frequency = random.random() * 30 + 20  # frequency
    phase = random.random() * 2 * math.pi  # phase
    rotation_count = random.randint(10, 20)  # of rotations

    for kw in range(width):
        y = float(kw) / (width - 1) * 4 * math.pi - 2 * math.pi
        for kh in range(height):
            x = float(kh) / (height - 1) * 4 * math.pi - 2 * math.pi
            z = 0.0
            for i in range(rotation_count):
                r = math.hypot(x, y)
                a = math.atan2(y, x) + i * math.pi * 2.0 / rotation_count
                z += math.cos(r * math.sin(a) * frequency + phase)
            c = int(255 - round(255 * z / rotation_count))
            pixels[kw, kh] = c  # grayscale
    return np.array(image.convert("RGB"))

def mono_back(height,width):
    back=np.zeros((height,width,3))
    back[:,:]=randColor()
    back=back.astype("uint8")
    return back

op_dict_back={}
op_dict_back["mono"]=mono_back
op_dict_back["paper"]=gaussian_noise
op_dict_back["crystal"]=quasicrystal

def get_background(img,back_paths):
    
    back_iden=random.choice(config.backs)
    if back_iden=="scene":
        back=get_scene_back(img,back_paths)
    else:
        back_funct=op_dict_back[back_iden]
        h,w=img.shape
        back=back_funct(h,w)
    return back

#----------------------------------------
# create foreground
#----------------------------------------
def get_foreground(img):
    fore_type=random.choice(["mono","color"])
    if fore_type=="color":
        h,w=img.shape
        fore=np.zeros((h,w,3))
        fore[:,:]=randColor()
    else:
        h,w=img.shape
        fore=np.zeros((h,w,3))
        fore[:,:]=randColor(col=False)

    img_r=255-img
    mask=cv2.merge((img_r,img_r,img_r))
    fore=0.1*mask+0.9*fore
    fore=fore.astype("uint8")
    fore[img==0]=(255,255,255)
    return fore
    
#----------------------------------------
# blending utils
#----------------------------------------
def solid_blend(back,fore,mask):
    fore[mask==0]=back[mask==0]
    return fore

def multi_blend(back,fore,blend):
    fore_b=cv2.cvtColor(fore,cv2.COLOR_RGB2RGBA)
    back_b=cv2.cvtColor(back,cv2.COLOR_RGB2RGBA)
    fore_b=fore_b.astype(float)
    back_b=back_b.astype(float)
    data=blend(back_b,fore_b,0.5)
    data=data.astype("uint8")
    data=cv2.cvtColor(data,cv2.COLOR_RGBA2RGB)
    return data

def check_visibility(image, mask):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    height, width = mask.shape

    peak = (mask > 127).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bound = (mask > 0).astype(np.uint8)
    bound = cv2.dilate(bound, kernel, iterations=1)

    visit = bound.copy()
    visit ^= 1
    visit = np.pad(visit, 1, constant_values=1)

    border = bound.copy()
    border[mask > 0] = 0

    flag = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    for y in range(height):
        for x in range(width):
            if peak[y][x]:
                cv2.floodFill(gray, visit, (x, y), 1, 16, 16, flag)

    visit = visit[1:-1, 1:-1]
    count = np.sum(visit & border)
    total = np.sum(border)
    return total > 0 and count <= total * 0.1

def get_blended_data(back,fore,mask):
    ops=["solid"]+BLENDS
    random.shuffle(ops)
    for op in ops:
        if op=="solid":
            data=solid_blend(back,fore,mask)
        else:
            data=multi_blend(back,fore,op)
        if check_visibility(data,mask):
            return data
    return fore 