#!/bin/sh
save_path="/backup2/STYLE_OCR/res/dataset/test/"
fonts_dir="/backup2/STYLE_OCR/res/fonts/"
backs_dir="/backup2/STYLE_OCR/res/backs/"
dict_path="/backup2/STYLE_OCR/res/srcs/bn_test.txt"
#--------------------------------------------------------------
python scene.py $save_path $fonts_dir $backs_dir $dict_path
#--------------------------------------------------------------
echo succeeded
