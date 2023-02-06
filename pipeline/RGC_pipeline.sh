# !/bin/bash

sorting_file_directory='/mnt/hdd1/RGC/sorting_files'
h5_files_directory='/mnt/hdd1/RGC/data'
stimulus_file_directory='/mnt/hdd1/RGC/data/stim'
bin_files_directory='/mnt/hdd2/RGC/bin_data'
kilosort_path='/home/amin/Documents/RGC/Kilosort'
npy_matlab_path=$kilosort_path'/npy-matlab'
path2cnfg='/home/amin/Documents/RGC/configFiles'
temporary_folder_directory='/home/amin/Documents/RGC/data/'

IFS=$'\n' 
declare -a routes=('20210108.json')
routes_path='['
for route_path in ${routes[@]}; 
    do
        routes_path=$routes_path$route_path','
    done
routes_path=${routes_path%?}']'
echo $routes_path
unset IFS

python3 developing/main.py --sorting_file_directory $sorting_file_directory --h5_files_directory $h5_files_directory --stimulus_file_directory  $stimulus_file_directory --bin_files_directory $bin_files_directory --routes $routes_path binary_conversion
python3 developing/main.py --sorting_file_directory $sorting_file_directory --h5_files_directory $h5_files_directory --stimulus_file_directory  $stimulus_file_directory --bin_files_directory $bin_files_directory --routes $routes_path mea_sort --kilosort_path $kilosort_path --npy_matlab_path $npy_matlab_path --path2cnfg $path2cnfg --temporary_folder_directory $temporary_folder_directory

