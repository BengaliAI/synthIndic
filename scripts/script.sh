#!/bin/sh
backs_dir="/home/ansary/WORK/Work/APSIS/datasets/SYNTH_INDIC/styles/"
bn_fonts="/home/ansary/WORK/Work/APSIS/datasets/SYNTH_INDIC/fonts/bangla/"
bn_std="/home/ansary/WORK/Work/APSIS/datasets/SYNTH_INDIC/fonts/bangla/Bangla.ttf"
en_fonts="/home/ansary/WORK/Work/APSIS/datasets/SYNTH_INDIC/fonts/english/"
en_std="/home/ansary/WORK/Work/APSIS/datasets/SYNTH_INDIC/fonts/english/English.ttf"
save_dir="/home/ansary/WORK/Work/APSIS/datasets/SYNTH_INDIC/datasets/"
#-----------------------------------synthetic------------------------------------------
python lang.py "bangla" "bn" $save_dir $bn_fonts $backs_dir $bn_std 
python lang.py "english" "en" $save_dir $en_fonts $backs_dir $en_std --num_samples 20000
#-----------------------------------synthetic------------------------------------------
echo succeeded