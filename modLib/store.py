# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import tensorflow as tf
from tqdm.auto import tqdm
tqdm.pandas()
import cv2
from .config import *
from .utils import *

FONT=None
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
                # image
                img=cv2.imread(img_path)
                img,_=correctPadding(img,(img_height,img_width))
                tmp_img_path=os.path.join(temp_path,f"img_{sidx}.png")
                cv2.imwrite(tmp_img_path,img)
                
                # target
                std=createFontImage(FONT,word)
                std,_=correctPadding(std,(img_height,img_width))
                tmp_tgt_path=os.path.join(temp_path,f"tgt_{sidx}.png")
                cv2.imwrite(tmp_tgt_path,std)
                
                # word
                word=[vocab.index(c) for c in word]
                for _ in range(pos_max-len(word)):
                    word+=[vocab.index('pad')]
                # img
                with(open(tmp_img_path,'rb')) as fid:
                    image_png_bytes=fid.read()
                
                with(open(tmp_tgt_path,'rb')) as fid:
                    target_png_bytes=fid.read()
                
                # feature desc
                data ={ 'image':_bytes_feature(image_png_bytes)}
                data['target']=_bytes_feature(target_png_bytes)
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

def createRecords(data,save_path,sidx,temp_path,font_path):
    global FONT
    FONT=PIL.ImageFont.truetype(font_path,comp_dim)
    print(f"Creating TFRECORDS:{save_path}")
    for idx in tqdm(range(0,len(data),tf_size)):
        df        =   data.iloc[idx:idx+tf_size] 
        df.reset_index(drop=True,inplace=True) 
        rnum      =   idx//tf_size
        toTfrecord(df,rnum,save_path,sidx,temp_path)

    
    