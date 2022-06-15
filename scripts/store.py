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
from tqdm.auto import tqdm
tqdm.pandas()
from rsLib.config import *
from rsLib.store import createRecords
from coreLib.utils import create_dir
from multiprocessing import Process
#------------------------
# data
#-------------------------

def main(args):
    save_path   =   args.save_path
    num_proc    =   int(args.num_process)
    data_csv    =   args.data_csv
    save_path   =   create_dir(save_path,"tfrecords")
    temp_path   =   create_dir(save_path,"temp")
    
    df=pd.read_csv(data_csv)
    dfs=[df[idx:idx+split] for idx in range(0,len(df),split)]
    max_end=len(dfs)


    def run(idx):
        if idx <len(dfs):
            tf_path=create_dir(save_path,str(idx))
            createRecords(dfs[idx],tf_path,idx,temp_path)


    def execute(start,end):
        process_list=[]
        for idx in range(start,end):
            p =  Process(target= run, args = [idx])
            p.start()
            process_list.append(p)
        for process in process_list:
            process.join()


    if max_end==1:
        dfs=[df]
        run(0)
    else:
        for i in range(0,max_end,num_proc):
            start=i
            end=start+num_proc
            if end>max_end:end=max_end-1
            execute(start,end) 

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Robust Scanner Dataset Creation Script")
    parser.add_argument("data_csv", help="Path of the data.csv file holding imagepath and word")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--num_process",required=False,default=16,help ="number of processes to be used:default=16")
    
    args = parser.parse_args()
    main(args)