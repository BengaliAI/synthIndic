# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import json
import math
import pandas as pd 
import tensorflow as tf
import numpy as np 
from ast import literal_eval
from tqdm.auto import tqdm
tqdm.pandas()
import cv2
from .config import *
#------------------------------------------------------------------------
def padWordImage(img,pad_loc,pad_dim,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_val :       the value to pad 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w,d=img.shape
        # pad widths
        pad_width =pad_dim-w
        # pads
        pad =np.ones((h,pad_width,3))*pad_val
        # pad
        img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w,d=img.shape
        # pad heights
        if h>= pad_dim:
            return img 
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8")    
#---------------------------------------------------------------
def correctPadding(img,dim,pvalue=255):
    '''
        corrects an image padding 
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w,d=img.shape
    
    w_new=int(img_height* w/h) 
    img=cv2.resize(img,(w_new,img_height))
    h,w,d=img.shape
    
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new))
        # pad
        img=padWordImage(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padWordImage(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_val=pvalue)
        mask=w
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height))
    
    return img,mask 
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def toTfrecord(df,rnum,rec_path,sidx,temp_path):
    '''
        args:
            df      :   the dataframe that contains the information to store
            rnum    :   record number
            rec_path:   save_path
            mask_dim:   the dimension of the mask
    '''
    tfrecord_name=f'{sidx}_{rnum}.tfrecord'
    tfrecord_path=os.path.join(rec_path,tfrecord_name) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        
        for idx in range(len(df)):
            try:
                img_path=df.iloc[idx,0]
                word=df.iloc[idx,1]
                img=cv2.imread(img_path)
                img,imask=correctPadding(img,(img_height,img_width))
                tmp_path=os.path.join(temp_path,f"{sidx}.png")
                cv2.imwrite(tmp_path,img)
                # mask
                imask=math.ceil((imask/img_width)*(img_width//factor))
                mask=np.zeros((img_height//factor,img_width//factor))
                mask[:,:imask]=1
                mask=mask.flatten().tolist()
                mask=[1-int(i) for i in mask]
                # word
                word=[vocab.index(c) for c in word]
                word=[vocab.index('sep')]+word+[vocab.index('sep')]
                for _ in range(pos_max-len(word)):
                    word+=[vocab.index('pad')]
                # img
                with(open(tmp_path,'rb')) as fid:
                    image_png_bytes=fid.read()
                # feature desc
                data ={ 'image':_bytes_feature(image_png_bytes)}
                data["mask"]=_int64_list_feature(mask)
                data["word"]=_int64_list_feature(word)
                
                features=tf.train.Features(feature=data)
                example= tf.train.Example(features=features)
                serialized=example.SerializeToString()
                writer.write(serialized)  
            except Exception as e:
                try:
                    print(img_path)
                    print(word)
                    print(e)
                except Exception as e2:
                    print(e2)

def createRecords(data,save_path,sidx,temp_path):
    print(f"Creating TFRECORDS:{save_path}")
    for idx in tqdm(range(0,len(data),tf_size)):
        df        =   data.iloc[idx:idx+tf_size] 
        df.reset_index(drop=True,inplace=True) 
        rnum      =   idx//tf_size
        toTfrecord(df,rnum,save_path,sidx,temp_path)

    
    