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
import pandas as pd
from tqdm import tqdm
from glob import glob 
from coreLib.scene import create_data
from coreLib.utils import LOG_INFO,create_dir
from multiprocessing import Process

#--------------------
# main
#--------------------


def main(args):
    save_path   =   args.save_path
    dict_txt    =   args.dict_txt
    num_proc    =   int(args.num_process)
    data_div    =   int(args.data_div)
    save_path   =   create_dir(save_path,"data")
    
    class resources:
        fonts   =   [ttf for ttf in tqdm(glob(os.path.join(args.fonts_dir,"*.ttf")))]
        backs   =   [back for back in tqdm(glob(os.path.join(args.backs_dir,"*.*")))]
    LOG_INFO(f"Resources: backs:{len(resources.backs)} fonts:{len(resources.fonts)}")
    # dictionary
    words=[]
    with open(dict_txt,"r") as f:
        _words=f.readlines()
    for word in tqdm(_words):
        if word.strip():
            words.append(word.strip())
    LOG_INFO(f"Number of data:{len(words)}")
    df=pd.DataFrame({"word":words})
    dicts=[]
    for idx in range(0,len(df),data_div):
        dicts.append(df[idx:idx+data_div])
    
    # ----------------------------------------------------------
    def _execute(idx):
        # data creation
        
        save_dir=create_dir(save_path,str(idx))
        create_data(save_dir,dicts[idx],resources)

    # ----------------------------------------------------------
    def run(start,end):
        process_list=[]
        for idx in range(start,end):
            p =  Process(target= _execute, args = [idx])
            p.start()
            process_list.append(p)
        for process in process_list:
            process.join()
    # ----------------------------------------------------------
    for i in tqdm(range(0,len(dicts),num_proc)):
        start=i
        end=start+num_proc
        if end>len(dicts):
            end=len(dicts)-1
        run(start,end)
    # ----------------------------------------------------------
    # clean-up and merging
    csvs=[]
    for i in tqdm(range(len(dicts))):
        csv=os.path.join(save_path,str(i),"data.csv")
        if os.path.exists(csv):
            csvs.append(csv)
    dfs=[pd.read_csv(csv) for csv in csvs]
    df=pd.concat(dfs,ignore_index=True)
    df.to_csv(os.path.join(save_path,"data.csv"),index=False)
    # temp files
    for csv in csvs:os.remove(csv)
        
        
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("SynthIndic recog Dataset Creating Script")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("fonts_dir", help="Path of the folder that contains fonts")
    parser.add_argument("backs_dir", help="Path of the folder that contains background images")
    parser.add_argument("dict_txt", help="Path of the dictionary txt to be used")
    
    parser.add_argument("--num_process",required=False,default=32,help ="number of processes to be used:default=32")
    parser.add_argument("--data_div",required=False,default=10000,help ="number of data to be used:default=10000")
    
    args = parser.parse_args()
    main(args)