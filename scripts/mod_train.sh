#!/bin/sh
save_path="/backup2/STYLE_OCR/res/dataset/train/"
fonts_dir="/backup2/STYLE_OCR/res/fonts/"
backs_dir="/backup2/STYLE_OCR/res/backs/"
dict_path="/backup2/STYLE_OCR/res/srcs/bn_train.txt"
font_path="/backup2/STYLE_OCR/res/srcs/bn.ttf"
data_csv="/backup2/STYLE_OCR/res/dataset/train/data/data.csv"
#--------------------------------------------------------------
python scene.py $save_path $fonts_dir $backs_dir $dict_path
python store_mod.py $data_csv $save_path $font_path
#--------------------------------------------------------------
echo succeeded