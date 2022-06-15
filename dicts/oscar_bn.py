# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------------------------
# imports
#--------------------------------------
import pandas as pd 
from tqdm import tqdm
from bnunicodenormalizer import Normalizer
from indicparser.langs import bangla
bangla.consonant_diacritics+=['ং','ঃ'] 
from indicparser import graphemeParser
from multiprocessing import Process

gp=graphemeParser("bangla")
bnorm=Normalizer()
tqdm.pandas()
#--------------------------------------
# globals
#--------------------------------------
numbers                =    ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯']
punctuations           =    ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '।']
data_div=10000
num_proc=20
# ----------------------------------------------------------
df=pd.read_csv("bn.csv")
dicts=[]
for idx in range(0,len(df),data_div):
    dicts.append(df[idx:idx+data_div])

# ----------------------------------------------------------
def execute(idx):
    df=dicts[idx]
    df["word"]=df["word"].progress_apply(lambda x: bnorm(x)["normalized"])
    df.dropna(inplace=True)
    df.to_csv(f"oscar/{idx}.csv",index=False)
# ----------------------------------------------------------
def run(start,end):
    process_list=[]
    for idx in range(start,end):
        p =  Process(target= execute, args = [idx])
        p.start()
        process_list.append(p)
    for process in process_list:
        process.join()
for i in tqdm(range(0,len(dicts),num_proc)):
    start=i
    end=start+num_proc
    if end>len(dicts):
        end=len(dicts)-1
    run(start,end)
