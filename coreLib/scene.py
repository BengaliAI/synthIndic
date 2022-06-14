# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import cv2
import numpy as np
import random
import pandas as pd 
from tqdm import tqdm
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont

from .style import get_background, get_blended_data, get_foreground 
from .augmentation import augment
from .config import config
from .utils import create_dir,LOG_INFO,random_exec,post_process_word_image

tqdm.pandas()

#-----------------------------------------
# data functions
#-----------------------------------------
def createFontImage(font,text):
    '''
        creates font-space target images
        args:
            font    :   the font to use
            comps   :   the list of graphemes
        return:
            non-pad-corrected raw binary target
    '''
    
    # draw text
    image = PIL.Image.new(mode='L', size=font.getsize(text))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text=text, fill=255, font=font)
    # clear extra white space
    img=np.array(image)
    idx=np.where(img>0)
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    img=img[y_min:y_max,x_min:x_max]
    return img    
    


def create_word_mask(word,font):
    # image
    img=createFontImage(font,word)
    img=post_process_word_image(img,config)
    img=np.squeeze(img)
    return img

def create_single_image(word,font,resources):
    mask=create_word_mask(word,font)
    # image
    back=get_background(mask,resources.backs)
    fore=get_foreground(mask)
    image=get_blended_data(back,fore,mask)
    image=augment(image,base=True)
    img_height=random.randint(config.min_dim,config.max_dim)
    h,w,_=image.shape
    w_new=int(img_height* w/h) 
    image=cv2.resize(image,(w_new,img_height))
    return image



#--------------------
# ops
#--------------------
def create_data(save_dir,dictionary,resources,comp_dim=64):
    '''
        creates: 
            * handwriten word image
            * fontspace word image
            * a dataframe/csv that holds word level groundtruth
    '''
    #---------------
    # processing
    #---------------
    # save_paths
    class save:    
        image=create_dir(save_dir,"image")
        csv=os.path.join(save_dir,"data.csv")
        txt=os.path.join(save_dir,"data.txt")
    
    filepaths=[]
    words=[]
    fiden=0
    # loop
    for idx in tqdm(range(len(dictionary))):
        try:
            word=dictionary.iloc[idx,0]
            font=PIL.ImageFont.truetype(random.choice(resources.fonts),comp_dim)
            # word mask
            image=create_single_image(word,font,resources)
            #-----------------------------------------------------------------------
            # save
            fname=f"{fiden}.png"
            cv2.imwrite(os.path.join(save.image,fname),image)
            filepaths.append(os.path.join(save.image,fname))
            words.append(word)
            fiden+=1
            with open(save.txt,"a+") as f:
                f.write(f"{fiden}.png  {word}\n")
        except Exception as e:
           LOG_INFO(e)
    
    df=pd.DataFrame({"filepath":filepaths,"word":words})
    df.to_csv(os.path.join(save.csv),index=False)