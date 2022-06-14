
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

# TODO
- [x] clean ops to reduce time
- [x] language : dict_grapheme from oscar corpus
    -   [x] lanuage agnostic
- [x] utils: geometric operation: rotation,warp
- [ ] ops: separate domain ops
- [ ] scene: stablize scene generation
- [x] augmentation: reduce augmentations with hyper-params
- [ ] paper: stablize paper generation
- [ ] scripts/scene.py: 
    - [ ] back dir
    - [ ] font dir
    - [ ] dictionary: complete
        - [ ] carefull construction for coverage 

```python
# scene config
class config:
    angle_max        = 5
    warping_max      = 10
    mask_neg         = 30    
    max_dim          = 128
    min_dim          = 16
    backs            = ["scene","mono"] # paper,crystal
# paper config
```