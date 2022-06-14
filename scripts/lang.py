#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
import os 
from tqdm import tqdm
from glob import glob 
from coreLib.data import createSyntheticData
from coreLib.languages import languages

#--------------------
# main
#--------------------
def main(args):
    language    =   args.language
    iden        =   args.iden
    save_path   =   args.save_path
    
    class resources:
        def_font=   args.standard_font
        fonts   =   [ttf for ttf in tqdm(glob(os.path.join(args.fonts_dir,"*.ttf")))]
        backs   =   [back for back in tqdm(glob(os.path.join(args.backs_dir,"*.*")))]
        
    
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    num_samples =   int(args.num_samples)
    img_dim=(img_height,img_width)
    language=languages[language]
    # data creation
    createSyntheticData(iden=iden,
                        save_dir=save_path,
                        language=language,
                        img_dim=img_dim,
                        resources=resources,
                        num_samples=num_samples)
    

    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Modifier Synthetic Dataset Creating Script")
    parser.add_argument("language", help="the specific language to use")
    parser.add_argument("iden",help="identifier to identify the dataset")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    
    parser.add_argument("fonts_dir", help="Path of the folder that contains fonts")
    parser.add_argument("backs_dir", help="Path of the folder that contains background images")
    parser.add_argument("standard_font", help="Path of the standard font to be used")
    
    
    parser.add_argument("--num_samples",required=False,default=100000,help ="number of samples to create when:default=100000")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width for each grapheme: default=512")
    
    
    args = parser.parse_args()
    main(args)