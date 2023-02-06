# !/bin/bash
file_path = '/mnt/hdd1/RGC/sorting_files/20210108.json'

matlab -nodisplay -r "cd('developing/'); MEA_sort3('$file_path');exit"
