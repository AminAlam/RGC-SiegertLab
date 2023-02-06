from __future__ import print_function, division

# major packages needed
import numpy as np
import h5py
import os
import sys
from scipy.io import savemat
import pickle as pkl
from utils import *


def binary_conversion_backend(ctx):
    sorting_file_directory = ctx.obj['sorting_file_directory']
    h5_files_directory = ctx.obj['h5_files_directory']
    stimulus_file_directory = ctx.obj['stimulus_file_directory']
    bin_files_directory = ctx.obj['bin_files_directory']
    routes = ctx.obj['routes']
    
    problematic_routes = []
    for route_name in routes:
        sorring_file_path = os.path.join(sorting_file_directory, route_name)
        paradigm, traces, folder_name, _ = load_sorting_data(sorring_file_path)
        print(traces)

        source = '%s/%s/'%(h5_files_directory, folder_name[0:4])
        
        if os.path.isdir(bin_files_directory+'/'+folder_name):
            print ('Output folder for %s exists'%folder_name)
        else:
            os.mkdir(bin_files_directory+'/'+folder_name)
            print ('Output folder for %s created'%folder_name)
                
        for trace_name in traces:
            filename = source+folder_name+'/'+trace_name
            print (filename)

            if os.path.isfile(filename):
                print ("File exist")
            else:
                print ("File not exist")
                continue

            if os.path.isfile("%s%s.bin"%(bin_files_directory+'/'+folder_name+'/',trace_name.rsplit(".",1)[0])):
                print("Binary file already there... Skipping to the next trace...")
                continue
            else:
                test_data = h5py.File(source+folder_name+'/'+trace_name, 'r')

                mapping = np.array(test_data['mapping'])
                c_coor = np.array([mapping[i][0] for i in np.arange(len(mapping))])
                k_coor = np.array([mapping[i][1] for i in np.arange(len(mapping))])
                x_coor = np.array([mapping[i][2] for i in np.arange(len(mapping))])
                y_coor = np.array([mapping[i][3] for i in np.arange(len(mapping))])

                c_coor = c_coor[np.argsort(k_coor)]-1

                try:
                    bin_filename = "%s%s.bin"%(bin_files_directory+'/'+folder_name+'/',trace_name.rsplit(".",1)[0])
                    data = np.memmap(bin_filename, dtype='uint16', mode='w+', shape=test_data['sig'].shape)
                    data[:] = test_data['sig']
                    data = data[c_coor.astype('int')].transpose()
                    data.astype('int16').tofile(bin_filename)
                    print('Data shape: ', data.shape)
                except:
                    print('Problem with converting %s'%(trace_name))
                    # problematic_routes.append(source+folder_name+'/'+trace_name)

        
        paradigm_list = ['MB', 'MS', 'S', 'N', 'G', 'FF', 'N', 'S', 'CH']
        
        stimulus = {}
        stimulus['times'] = {}
        stimulus['bits'] = {}
        stimulus['length'] = {}
        for i in np.arange(len(traces)):
            test_data = h5py.File(source+folder_name+'/'+traces[i], 'r')

            X = np.array([test_data['sig'][1026][0], test_data['sig'][1027][0]])
            t_offset = (X[1]<<16)|X[0]

            stimulus['times'][paradigm[i]] = test_data['bits']['frameno']-t_offset
            stimulus['bits'][paradigm[i]] = test_data['bits']['bits']
            stimulus['length'][paradigm[i]] = test_data['sig'].shape[1]
            
        save_obj(stimulus, '%s/%s'%(stimulus_file_directory,route_name))

    print("Conversion done!")

    # problem_filename = '%s/%s'%(stimulus_file_directory,route_name)
    #np.savetxt(problematic_routes, problem_filename)
    # print('Problematic routes stored in %s'%problem_filename)
