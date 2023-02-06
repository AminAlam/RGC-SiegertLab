import pandas as pd
import numpy as np
import h5py
import pickle as pkl
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy.stats import gaussian_kde
from scipy.signal import convolve
from utils import *
import json

#### PATHS
archive_path = '/Volumes/malamalh/archive-siegegrp' #'/home/malamalh/mounts/drives/archive-siegegrp'
fs3_path = '/Volumes/malamalh/fs3-siegegrp' #'/home/malamalh/mounts/drives/fs3-siegegrp'
sorting_file_directory = f'{archive_path}/Natalie/ks3_spikesorting/ks_input/txt_sorting_files/Sorting/'
kilo_sort_files_parent_path = f'{archive_path}/RGC_Sorting/Results/'
stimulus_files_path = f'{fs3_path}/Ryan/RGC/MEA_Timing'
stimulus_pickle_bool = False 
#### PATHS


#### Inputs
route_list = ['20200419_Route1307']
paradigm_list = ['MB0']
#### Inputs


#### Parameters
sampling_rate = 20000.
fr_cutoff = 1.50 # in seconds
window_length = 20
window = np.arange(window_length) - (window_length/2)
sdev = 10
window = np.exp(-(window**2)/(2*(sdev**2)))/(np.sqrt(2*np.pi*sdev))
window_cutoff = 20
#### Parameters


def stimulus_dict_initiating(traces, paradigm, fnames):
    stimulus = {}
    stimulus['times'] = {}
    stimulus['bits'] = {}
    stimulus['length'] = {}

    for i in np.arange(len(traces)):
        # print(traces[i], paradigm[i])
        if traces[i].split('.')[-1]=='h5':
            if stimulus_pickle_bool:
                stimulus_file_name = f'{stimulus_files_path}/{fnames}__{traces[i].split(".")[0]}.pkl'
                stimulus = load_obj(stimulus_file_name)
                t_offset = np.array(file['stimulus']['t_offset']).flatten()[0]
                stimulus['length'][paradigm[i]+str(i)] = np.array(file['stimulus']['length']).flatten()[0]
                stimulus['times'][paradigm[i]+str(i)] =  np.array(file['stimulus']['time']).flatten()
                stimulus['bits'][paradigm[i]+str(i)] =  np.array(file['stimulus']['bits']).flatten()
            else:
                stimulus_file_name = f'{stimulus_files_path}/{fnames}__{traces[i].split(".")[0]}.mat'
                with h5py.File(stimulus_file_name, 'r') as file:
                    t_offset = np.array(file['stimulus']['t_offset']).flatten()[0]
                    stimulus['length'][paradigm[i]+str(i)] = np.array(file['stimulus']['length']).flatten()[0]
                    stimulus['times'][paradigm[i]+str(i)] =  np.array(file['stimulus']['time']).flatten()
                    stimulus['bits'][paradigm[i]+str(i)] =  np.array(file['stimulus']['bits']).flatten()
        else:
            stimulus['length'][paradigm[i]+str(i)] = np.array([])
            stimulus['times'][paradigm[i]+str(i)] =  np.array([])
            stimulus['bits'][paradigm[i]+str(i)] =  np.array([])
    stimulus['paradigms'] = paradigm

    # disentangle stimulus paradigm spikes
    stimulus['time_range'] = {}
    stimulus['spike_times'] = {}
    stimulus['spike_clusters'] = {}
    stimulus['light_stimulus'] = {}
    stimulus['dark_stimulus'] = {}

    return stimulus

def spike_sort_info_files(kilo_sort_files_path):
    spike_times = np.load(f'{kilo_sort_files_path}spike_times.npy')
    spike_clusters = np.load(f'{kilo_sort_files_path}spike_templates.npy').flatten()
    templates = np.load(f'{kilo_sort_files_path}templates.npy')
    whitening_mat = np.load(f'{kilo_sort_files_path}whitening_mat.npy')
    channel_map = np.load(f'{kilo_sort_files_path}channel_map.npy')
    channel_positions = np.load(f'{kilo_sort_files_path}channel_positions.npy')

    return spike_times, spike_clusters, templates, whitening_mat, channel_map, channel_positions

def marching_square(stimulus):
    bits = stimulus['bits']['MS1'].flatten()
    time = stimulus['times']['MS1'].flatten()

    ts = time[np.where(bits==1)[0]][1:-1]
    idx = np.diff(ts)>5000
    t_start = np.hstack([ts[0], ts[np.where(idx)[0]+1]])
    t_stop = np.hstack([ts[np.where(idx)[0]], ts[-1]])

    light_stimulus = np.vstack((t_start,t_stop)).T

    dt = np.round(np.mean(np.diff(light_stimulus, axis=1))).astype('int')
    dark_stimulus = np.vstack((light_stimulus[:,1], np.hstack([light_stimulus[:,0][1:], light_stimulus[:,1][-1]+dt]))).T

    stimulus['light_stimulus']['MS1'] = light_stimulus
    stimulus['dark_stimulus']['MS1'] = dark_stimulus

    print('Marching square number of trials (must be 150): %d'%light_stimulus.shape[0])
    print('Average light stimulation duration: %.3f'%(np.mean(light_stimulus[:,1]-light_stimulus[:,0])/sampling_rate))
    print('Average dark stimulation duration: %.3f'%(np.mean(dark_stimulus[:,1]-dark_stimulus[:,0])/sampling_rate))
    
    MS_time = np.mean(light_stimulus[:,1]-light_stimulus[:,0])/sampling_rate

    return stimulus

def moving_bars(stimulus):
    bits = stimulus['bits']['MB0'].flatten()
    time = stimulus['times']['MB0'].flatten()

    if 27 in bits: ts = time[np.where(bits==27)[0]]
    else: ts = time[np.where(bits==0)[0]]
        
    ts2 = time[np.where(bits==1)[0]]
    idx = np.diff(ts)>120000
    idx2 = np.diff(ts2)>20000
    start_points = np.hstack([ts[0], ts[np.where(idx)[0]+1], ts2[np.where(idx2)[0]+1]])

    t_start = start_points[:-1]
    t_stop = start_points[1:]

    light_stimulus = np.vstack((t_start,t_stop)).T
    dark_stimulus = []
    
    stimulus['light_stimulus']['MB0'] = light_stimulus
    stimulus['dark_stimulus']['MB0'] = dark_stimulus
    
    print('Moving bar number of trials (must be 48): %d'%light_stimulus.shape[0])
    print('Average moving bar stimulation duration: %.3f'%(np.mean(light_stimulus[:,1]-light_stimulus[:,0])/sampling_rate))

    return stimulus

def calc_bias_index(stimulus, neuron_list, window, window_cutoff):
    resps = np.array([stimulus['spike_times']['MS1'], stimulus['spike_clusters']['MS1']]).T
    resps[:,0] = resps[:,0] - stimulus['time_range']['MS1'][0]
    cell_index = np.sort(neuron_list)

    spike_light = {}; spike_dark = {}
    fr_light = {}; fr_dark = {}
    psth_light = {}; psth_dark = {}
    psth_lat_light = {}; psth_lat_dark = {}
    sp_light_t = {}; sp_dark_t = {}

    
    # grid_arr = np.linspace(0,fr_cutoff,1000)
    grid_arr = np.arange(0,1.50+0.010,0.001)
    gaussian_width = 0.005

    

    lat_cutoff = 0.5 # in seconds
    lat_arr = np.arange(0,1.00+0.010,0.001)
    

    light_stimulus = stimulus['light_stimulus']['MS1']
    dark_stimulus = stimulus['dark_stimulus']['MS1']

    print('Preparing marching square responses...')
    total_spike = []
    for ind in np.arange(len(neuron_list)):
        spike_t = resps[:,0][np.where(resps[:,1]==neuron_list[ind])]
        total_spike.append(len(spike_t))

        spike_light[ind] = {}
        spike_dark[ind] = {}

        psth_light[ind] = {}
        psth_dark[ind] = {}

        fr_light[neuron_list[ind]] = {}; fr_dark[neuron_list[ind]] = {}
        sp_light_t[neuron_list[ind]] = {}; sp_dark_t[neuron_list[ind]] = {}
            
        for pix in np.arange(25):
            loc_pix = np.array([i for i in np.arange(len(light_stimulus)) if np.mod(i,25)==pix])

            light_t = light_stimulus[loc_pix]
            dark_t = dark_stimulus[loc_pix]

            spike_light[ind][pix] = {}; spike_dark[ind][pix] = {}
            psth_lat_light[ind] = {}; psth_lat_dark[ind] = {}

            fr_l = []; fr_d = []
            sp_light = []; sp_dark = []
            full_light = []; full_dark = []
            sp_lat_light = []; sp_lat_dark = []            
            
            
            for t in np.arange(1, len(light_t)):
                spike_light[ind][pix][t] = spike_t[np.where((spike_t>=light_t[t][0])*(spike_t<light_t[t][0]+fr_cutoff*sampling_rate))[0]] - light_t[t][0]
                spike_dark[ind][pix][t] = spike_t[np.where((spike_t>=dark_t[t][0])*(spike_t<dark_t[t][0]+fr_cutoff*sampling_rate))[0]] - dark_t[t][0]

                sp_light.append(spike_t[np.where((spike_t>=light_t[t][0])*(spike_t<light_t[t][0]+fr_cutoff*sampling_rate))[0]] - light_t[t][0])
                sp_dark.append(spike_t[np.where((spike_t>=dark_t[t][0])*(spike_t<dark_t[t][0]+fr_cutoff*sampling_rate))[0]] - dark_t[t][0])

                sp_lat_light.append(spike_t[np.where((spike_t>=light_t[t][0])*(spike_t<light_t[t][0]+lat_cutoff*sampling_rate))[0]] - light_t[t][0])
                sp_lat_dark.append(spike_t[np.where((spike_t>=dark_t[t][0])*(spike_t<dark_t[t][0]+lat_cutoff*sampling_rate))[0]] - dark_t[t][0])

            # print(len(sp_light), len(sp_dark))
            sp_light = np.sort(np.hstack(np.array(sp_light))/sampling_rate)
            sp_dark = np.sort(np.hstack(np.array(sp_dark))/sampling_rate)

            sp_lat_light = np.sort(np.hstack(np.array(sp_lat_light))/sampling_rate)
            sp_lat_dark = np.sort(np.hstack(np.array(sp_lat_dark))/sampling_rate)
            
            fr_light_t = np.histogram(sp_light, grid_arr)[0]
            convolved_spike = convolve(fr_light_t, window, mode='same')
            convolved_spike[0:int(window_cutoff)] = 0
            convolved_spike[-int(window_cutoff):] = 0
            psth_light[ind][pix] = convolved_spike
        
            fr_dark_t = np.histogram(sp_dark, grid_arr)[0]
            convolved_spike = convolve(fr_dark_t, window, mode='same')
            convolved_spike[0:int(10)] = 0
            convolved_spike[-int(10):] = 0
            psth_dark[ind][pix] = convolved_spike

            fr_light[neuron_list[ind]][pix] = len(sp_light)
            fr_dark[neuron_list[ind]][pix] = len(sp_dark)

            sp_light_t[neuron_list[ind]][pix] = sp_light
            sp_dark_t[neuron_list[ind]][pix] = sp_dark

    print('Calculating bias indices...')
    bias = np.zeros(len(cell_index))
    latency = np.zeros(len(cell_index))
    transience = np.zeros(len(cell_index))
    surround_unit = np.zeros(len(cell_index))

    for ind in np.arange(len(cell_index)):
        # generate the peak firing for each square
        max_fr_light_values = np.array([np.amax(psth_light[ind][pix]) for pix in np.arange(25)])
        max_fr_dark_values = np.array([np.amax(psth_dark[ind][pix]) for pix in np.arange(25)])
        # find the square with the largest peak firing
        if np.amax(max_fr_light_values) >= np.amax(max_fr_dark_values): max_fr_index = np.argmax(max_fr_light_values)
        else: max_fr_index = np.argmax(max_fr_dark_values)

        mod_d, mod_r = int(max_fr_index/5), int(max_fr_index%5)
        # get all the surrounding squares, note if it is an edge
        surround_index_d = np.array([ mod_d-1, mod_d-1, mod_d-1, mod_d, mod_d, mod_d, mod_d+1, mod_d+1, mod_d+1 ]).astype('int')
        surround_index_r = np.array([ mod_r-1, mod_r, mod_r+1, mod_r-1, mod_r, mod_r+1, mod_r-1, mod_r, mod_r+1 ]).astype('int')
        surround_index = np.array([ 5*surround_index_d[xx]+surround_index_r[xx] for xx in np.arange(len(surround_index_d)) if ((surround_index_d[xx] not in [-1,5]) and (surround_index_r[xx] not in [-1,5])) ]).astype('int')
        if len(surround_index)<9: surround_unit[ind] = 1.0
        
        max_fr_light = max_fr_light_values[max_fr_index]
        max_fr_dark = max_fr_dark_values[max_fr_index]
        
        preferred_light_pix = np.argmax(np.array([np.amax(psth_light[ind][pix]) for pix in np.arange(25)]))
        preferred_dark_pix = np.argmax(np.array([np.amax(psth_dark[ind][pix]) for pix in np.arange(25)]))
                
        if (max_fr_light+max_fr_dark)>0:
            bias[ind] = (max_fr_light - max_fr_dark)/(max_fr_light + max_fr_dark)
            if max_fr_light >= max_fr_dark:
                latency[ind] = np.argmax(psth_light[ind][preferred_light_pix]+1)*np.diff(lat_arr)[0]
                transience[ind] = np.sum(psth_light[ind][preferred_light_pix]/np.amax(psth_light[ind][preferred_light_pix]))*(np.diff(lat_arr)[0]/1.5)
            else:
                latency[ind] = np.argmax(psth_dark[ind][preferred_dark_pix]+1)*np.diff(lat_arr)[0]
                transience[ind] = np.sum(psth_dark[ind][preferred_dark_pix]/np.amax(psth_dark[ind][preferred_dark_pix]))*(np.diff(lat_arr)[0]/1.5)
        else:
            bias[ind] = np.nan
            latency[ind] = np.nan
            transience[ind] = np.nan
    
    stimulus['bias_index'] = bias
    stimulus['latency_index'] = latency
    stimulus['transience_index'] = transience
    stimulus['MS_total_spike'] = total_spike
    stimulus['surround_unit'] = surround_unit
    print('Complete!')
    return stimulus, cell_index, psth_light, psth_dark, fr_light, fr_dark, sp_light_t, sp_dark_t

def calc_spontaneous_firing(stimulus, neuron_list, window, window_cutoff):
    print('Calculating spontaneous firing...')
    spontaneous_firing = np.zeros_like(neuron_list).astype('float')
    t_start = stimulus['light_stimulus']['MB0'][0][0]
    MB_time = t_start/sampling_rate
    MB_grid_arr = np.arange(0,MB_time+0.010,0.001)
    for i in np.arange(len(neuron_list)):
        neuron_index = np.where(stimulus['spike_clusters']['MB0']==neuron_list[i])[0]
        if len(neuron_index)>0:
            t_spikes = stimulus['spike_times']['MB0'][neuron_index] - stimulus['time_range']['MB0'][0]
            spontanrous_spikes = np.histogram(t_spikes[np.where(t_spikes<=t_start)[0]]/sampling_rate,MB_grid_arr)[0]
            convolved_spike = convolve(spontanrous_spikes, window, mode='same')
            convolved_spike[0:int(10)] = 0
            convolved_spike[-int(10):] = 0
            spontaneous_firing[i] = np.amax(convolved_spike)
    stimulus['spontaneous_firing'] = spontaneous_firing.astype('float')
    print('Complete!')
    return stimulus

def calc_responsiveness(stimulus, cell_index, psth_light, psth_dark):
    print('Calculating responsiveness...')
    responsive = np.zeros_like(cell_index)
    max_firing = np.zeros_like(cell_index).astype('float')
    rate_fc = np.zeros_like(cell_index).astype('float')

    for ind in np.arange(len(cell_index)):
        max_fr_light = np.array([np.amax(psth_light[ind][pix]) for pix in np.arange(25)])
        max_fr_dark = np.array([np.amax(psth_dark[ind][pix]) for pix in np.arange(25)])
        max_fr = np.amax([np.amax(max_fr_light), np.amax(max_fr_dark)])
        max_firing[ind] = max_fr

        if stimulus['spontaneous_firing'][ind] > 0:
            if max_fr > 0 :
                foldchange = max_fr/stimulus['spontaneous_firing'][ind]
            if max_fr == 0:
                foldchange = 0
        else:
            if max_fr > 0 : 
                foldchange = 10000
            if max_fr == 0:
                foldchange = 0

        rate_fc[ind] = foldchange

        if (foldchange>1.5) and (max_fr>0):
            responsive[ind] = 1.0

    stimulus['responsive'] = responsive
    stimulus['max_firing'] = max_firing
    stimulus['rate_fc'] = rate_fc
    
    print('Number of neurons: %d'%(len(responsive)))
    print('Number of responsive neurons: %d'%(np.sum(responsive)))
    print('Complete!')

    return stimulus, responsive

def calc_directional_selectivity(stimulus, neuron_list, window):
    print('Calculating direction selectivity and orientation selectivity index from moving bar...')
    resps = np.array([stimulus['spike_times']['MB0'], stimulus['spike_clusters']['MB0']]).T
    resps[:,0] = resps[:,0] - stimulus['time_range']['MB0'][0]
    light_stimulus = stimulus['light_stimulus']['MB0']
    
    MB_duration = np.mean(light_stimulus[:,1]-light_stimulus[:,0])/sampling_rate
    MB_grid_arr = np.arange(0,MB_duration+0.010,0.001)

    directions = np.array([270, 225, 180, 135, 90, 45, 0, 315]*6)
    unique_dirs = np.unique(directions)
    unique_dirs_rad = np.deg2rad(unique_dirs)
    unique_os_dirs_rad = np.deg2rad(2*unique_dirs)

    DS_index = np.zeros(len(neuron_list))
    OS_index = np.zeros(len(neuron_list))

    total_spike = []
    spike_times_th = {}
    firing_rates = {}
    psth_MB = {}
    for ind in np.arange(len(neuron_list)):
        spike_t = resps[:,0][np.where(resps[:,1]==neuron_list[ind])]
        total_spike.append(len(spike_t))

        mean_fr = {}    
        spike_times_th[neuron_list[ind]] = {}
        psth_MB[neuron_list[ind]] = {}
        for degs in unique_dirs:
            loc_degs = np.array([i for i in np.arange(len(directions)) if directions[i]==degs])
            deg_t = light_stimulus[loc_degs]
            mean_fr[degs] = []
            spike_times_th[neuron_list[ind]][degs] = []
            for t in np.arange(1, len(deg_t)):
                bar_duration = (deg_t[t][1] - deg_t[t][0])/sampling_rate
                sp_light = spike_t[np.where((spike_t>=deg_t[t][0])*(spike_t<deg_t[t][1]))[0]] - deg_t[t][0]
                mean_fr[degs].append(len(sp_light)/bar_duration)
                spike_times_th[neuron_list[ind]][degs].append(sp_light/sampling_rate)
            
            MB_fr_light_t = np.histogram(np.hstack(spike_times_th[neuron_list[ind]][degs]),MB_grid_arr)[0]
            convolved_spike = convolve(MB_fr_light_t, window, mode='same')
            convolved_spike[0:int(10)] = 0
            convolved_spike[-int(10):] = 0
            psth_MB[neuron_list[ind]][degs] = convolved_spike

        ds_array = np.zeros((5, len(unique_dirs)))
        for k in np.arange(len(unique_dirs)):
            ds_array[:, k] = mean_fr[unique_dirs[k]]
        ds_mean = np.sum(ds_array, axis=0)
        firing_rates[neuron_list[ind]] = ds_mean
        ds_mean = ds_mean/np.sum(ds_mean)
        
        ds_max = np.zeros(len(unique_dirs))
        for k in np.arange(len(unique_dirs)):
            ds_max[k] = np.amax(psth_MB[neuron_list[ind]][unique_dirs[k]])
            
        if len(spike_t) >= 50:
            DS_index[ind] = np.sqrt(np.power(np.sum(ds_mean*np.cos(unique_dirs_rad)),2) + np.power(np.sum(ds_mean*np.sin(unique_dirs_rad)), 2))
            OS_index[ind] = np.sqrt(np.power(np.sum(ds_mean*np.cos(unique_os_dirs_rad)),2) + np.power(np.sum(ds_mean*np.sin(unique_os_dirs_rad)), 2))
        else:
            DS_index[ind] = np.nan
            OS_index[ind] = np.nan

    stimulus['MB_DS_index'] = DS_index
    stimulus['MB_OS_index'] = OS_index
    stimulus['MB_total_spike'] = np.array(total_spike)

    print('Complete!')

    return stimulus, unique_dirs_rad, unique_os_dirs_rad, firing_rates, psth_MB, spike_times_th

def make_figure(fnames, route_name, neuron_ind, responsive, ind, templates, whitening_mat, channel_positions, sp_light_t, sp_dark_t,
                fr_light, fr_dark, unique_dirs_rad, firing_rates, spike_times_th, figure_save_path):

    fig = plt.figure(dpi=300)
    fig.set_size_inches(20,6)

    fig.suptitle('%s-%s-Unit%d (Responsive: %d, Good unit: )'%(fnames, route_name, neuron_ind, responsive[ind]))

    gs0 = gridspec.GridSpec(1, 3, figure=fig)

    # waveform templates
    n_channels = templates[neuron_ind].shape[1]

    offsets = np.array([ (-35.0, 35.0),  (-17.5, 35.0) , (0, 35.0),  (17.5, 35.0),  (35.0, 35.0),
                            (-35.0, 17.5),  (-17.5, 17.5) , (0, 17.5),  (17.5, 17.5),  (35.0, 17.5),
                            (-35.0, 0),     (-17.5, 0),     (0, 0),     (17.5, 0),     (35.0, 0),
                            (-35.0, -17.5), (-17.5, -17.5), (0, -17.5), (17.5, -17.5), (35.0, -17.5),
                            (-35.0, -35.0), (-17.5, -35.0), (0, -35.0), (17.5, -35.0), (35.0, -35.0)])

    temps = templates[neuron_ind]@whitening_mat
    amplitude = [(np.amax(np.abs(temps[:,chan])) - np.amin(np.abs(temps[:,chan]))) for chan in np.arange(n_channels)]
    max_amp = np.argmax(amplitude)
    ymax = np.amax([np.amax(temps[:,chan]) for chan in np.arange(n_channels)])
    ymin = np.amin([np.amin(temps[:,chan]) for chan in np.arange(n_channels)])

    max_chan_pos = channel_positions[max_amp]
    channel_offsets = max_chan_pos+offsets
    offset_index = [np.where((channel_positions[:,0]==channel_offsets[kk][0])*(channel_positions[:,1]==channel_offsets[kk][1]))[0] for kk in np.arange(len(channel_offsets))]

    gs00 = gridspec.GridSpecFromSubplotSpec(5, 5, wspace=0.5, hspace=0.5, subplot_spec=gs0[0])
    for kk in np.arange(len(offset_index)):
        ax = fig.add_subplot(gs00[int(kk/5),np.mod(kk,5)])
        if len(offset_index[kk]) > 0:
            ax.plot(temps[:,offset_index[kk][0]], 'k')
            ax.set_ylim(top=ymax, bottom=ymin)
        ax.axis('off')


    # bias index
    gs11 = gridspec.GridSpecFromSubplotSpec(2, 2, height_ratios=[2,1], hspace=0.3, subplot_spec=gs0[1])

    ax1 = fig.add_subplot(gs11[0,0])
    spike_times = [sp_light_t[neuron_ind][pix] for pix in np.arange(25)]
    ax1.eventplot(spike_times, linelengths=0.8, color='k', alpha=0.7)
    ax1.set_xlim(left=0, right=fr_cutoff)
    ax1.set_title('Light on')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('pixel index')

    ax2 = fig.add_subplot(gs11[0,1])
    spike_times = [sp_dark_t[neuron_ind][pix] for pix in np.arange(25)]
    ax2.eventplot(spike_times, linelengths=0.8, color='k', alpha=0.7)
    ax2.set_xlim(left=0, right=fr_cutoff)
    ax2.set_title('Light off')
    ax2.set_xlabel('time (s)')

    ax3 = fig.add_subplot(gs11[1,0])
    fr_pixel = np.array([fr_light[neuron_ind][pix] for pix in np.arange(25)]).reshape((5,5))
    im = ax3.imshow(np.rot90(fr_pixel), cmap=cm.afmhot)
    fig.colorbar(im, ax=ax3)

    ax4 = fig.add_subplot(gs11[1,1])
    fr_pixel = np.array([fr_dark[neuron_ind][pix] for pix in np.arange(25)]).reshape((5,5))
    im = ax4.imshow(np.rot90(fr_pixel), cmap=cm.afmhot)
    fig.colorbar(im, ax=ax4)


    # direction selectivity
    gs22 = gridspec.GridSpecFromSubplotSpec(3, 3, wspace=0.3, hspace=0.3, subplot_spec=gs0[2])

    ax = fig.add_subplot(gs22[1,1], projection='polar')
    ax.bar(unique_dirs_rad, firing_rates[neuron_ind], width=0.5, alpha=0.6)
    ax.grid(False)

    ax = fig.add_subplot(gs22[1,2]) #0 degrees
    ax.eventplot(spike_times_th[neuron_ind][0], linelengths=0.6, color='k', alpha=0.7)

    ax = fig.add_subplot(gs22[0,2]) #45 degrees
    ax.eventplot(spike_times_th[neuron_ind][45], linelengths=0.8, color='k', alpha=0.7)

    ax = fig.add_subplot(gs22[0,1]) #90 degrees
    ax.eventplot(spike_times_th[neuron_ind][90], linelengths=0.8, color='k', alpha=0.7)

    ax = fig.add_subplot(gs22[0,0]) #135 degrees
    ax.eventplot(spike_times_th[neuron_ind][135], linelengths=0.8, color='k', alpha=0.7)

    ax = fig.add_subplot(gs22[1,0]) #180 degrees
    ax.eventplot(spike_times_th[neuron_ind][180], linelengths=0.8, color='k', alpha=0.7)

    ax = fig.add_subplot(gs22[2,0]) #225 degrees
    ax.eventplot(spike_times_th[neuron_ind][225], linelengths=0.8, color='k', alpha=0.7)

    ax = fig.add_subplot(gs22[2,1]) #270 degrees
    ax.eventplot(spike_times_th[neuron_ind][270], linelengths=0.8, color='k', alpha=0.7)

    ax = fig.add_subplot(gs22[2,2]) #315 degrees
    ax.eventplot(spike_times_th[neuron_ind][315], linelengths=0.8, color='k', alpha=0.7)

    plt.savefig(figure_save_path, dpi=400)
    plt.close()

def calculate_stimulus_metrics(route_name, paradigms_of_interest):

    sorting_file_path = f"{sorting_file_directory}{route_name}.json"
    paradigm, traces, experiment_name, fnames = load_sorting_data(sorting_file_path)
    print('Loading traces for %s...'%fnames)
    kilo_sort_files_path = f'{kilo_sort_files_parent_path}{fnames}/{route_name}/'
    spike_times, spike_clusters, templates, whitening_mat, channel_map, channel_positions = spike_sort_info_files(kilo_sort_files_path)
    # initialize the stimulus dictionary
    stimulus = stimulus_dict_initiating(traces, paradigm, fnames)

    parad_idx = {}
    for xx_interest in paradigms_of_interest:
        parad_idx[xx_interest] = np.where(paradigm==xx_interest)[0]
        print(xx_interest, parad_idx[xx_interest])

    total_time = 0
    for i in range(len(paradigm)):
        start_time = total_time
        total_time += stimulus['length'][paradigm[i]+str(i)]
        stimulus['time_range'][paradigm[i]+str(i)] = np.array([start_time, total_time])

        temp_index = ((spike_times>=stimulus['time_range'][paradigm[i]+str(i)][0]) * (spike_times<stimulus['time_range'][paradigm[i]+str(i)][1]))
        stimulus['spike_times'][paradigm[i]+str(i)] = spike_times[temp_index]
        stimulus['spike_clusters'][paradigm[i]+str(i)] = spike_clusters[temp_index.flatten()]

    neuron_list = np.unique(spike_clusters)
    stimulus['neurons'] = neuron_list

    print('There are %d neurons...'%len(neuron_list))
    print('Disentangling stimulus and responses...')
    
    # marching square stimulation
    stimulus = marching_square(stimulus)

    # moving bars stimulation
    stimulus = moving_bars(stimulus)
    
    # calculate the bias from the marching square paradigm
    stimulus, cell_index, psth_light, psth_dark, fr_light, fr_dark, sp_light_t, sp_dark_t = calc_bias_index(stimulus, neuron_list, window, window_cutoff)

    # calculate spontaneous firing
    stimulus = calc_spontaneous_firing(stimulus, neuron_list, window, window_cutoff)
    
    # calculate responsiveness
    stimulus, responsive = calc_responsiveness(stimulus, cell_index, psth_light, psth_dark)
    
    # let's estimate the directional selectivity from moving bars
    stimulus, unique_dirs_rad, unique_os_dirs_rad, firing_rates, psth_MB, spike_times_th = calc_directional_selectivity(stimulus, neuron_list, window)
    
    # machine_predict = load_obj(f'{fs3_path}/Ryan/RGC/Updated_MEA/Classification/predicted_classification_20210513')
    # neuron_listing = np.arange(len(neuron_list))
    # indices = np.array([np.where(neuron_list==neuron)[0][0] for neuron in machine_predict[route_name].keys() if machine_predict[route_name][neuron] == 1]).flatten()
    # indices = np.array(1)
    # y_machine = np.zeros(len(neuron_listing))
    # y_machine[indices] = 1
    # stimulus['prediction'] = np.array(y_machine)
    # print('Percent of good neurons: %.2f'%np.mean(y_machine))

    
    print('Saving file...')
    save_obj(stimulus, f'{fs3_path}/Ryan/RGC/MEA_stimulus/Parameters_MBMS/%s_complete_MBMS_20210513'%(route_name))
    print('Complete!')


    if not os.path.exists(f'{kilo_sort_files_path}Waveforms'):
        os.makedirs(f'{kilo_sort_files_path}Waveforms')

    for ind in np.arange(len(neuron_list)):
        neuron_ind = neuron_list[ind]
        print('Creating figure for Unit %d...'%neuron_ind)
        figure_save_path = f'{kilo_sort_files_path}Waveforms/newest_%s-%s-Unit%d.png'%(fnames, route_name, neuron_ind)
        make_figure(fnames, route_name, neuron_ind, responsive, ind, templates, whitening_mat, channel_positions, sp_light_t, sp_dark_t,
                    fr_light, fr_dark, unique_dirs_rad, firing_rates, spike_times_th, figure_save_path)
        

if __name__ == '__main__':
    for k in np.arange(len(route_list)):
        calculate_stimulus_metrics(route_list[k], paradigm_list[k])