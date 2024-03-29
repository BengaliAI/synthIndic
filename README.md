
# SYNTH-INDIC 

```python
Version: 0.0.2
```

**LOCAL ENVIRONMENT**  

```python
OS          : Ubuntu 20.04.3 LTS       
Memory      : 23.4 GiB 
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.36.8
```

# Environment Setup

>Assuming the **libraqm** complex layout is working properly, you can skip to **python requirements**. 

* Install libraqm as described [here](https://github.com/HOST-Oman/libraqm)
* ```sudo ldconfig``` (librarqm local repo)

**python requirements**

* **pip requirements**: ```pip install -r requirements.txt``` 

> Its better to use a virtual environment 
> OR use conda-

* **conda**: use environment.yml: ```conda env create -f environment.yml```


# Execution
* change directory: ```cd scripts```
* run ```scene.py```

```python
usage: SynthIndic recog Dataset Creating Script [-h] [--num_process NUM_PROCESS] [--data_div DATA_DIV] save_path fonts_dir backs_dir dict_txt

positional arguments:
  save_path             Path of the directory to save the dataset
  fonts_dir             Path of the folder that contains fonts
  backs_dir             Path of the folder that contains background images
  dict_txt              Path of the dictionary txt to be used

optional arguments:
  -h, --help            show this help message and exit
  --num_process NUM_PROCESS
                        number of processes to be used:default=24
  --data_div DATA_DIV   number of data to be used:default=10000

```