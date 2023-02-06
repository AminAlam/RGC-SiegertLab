function MEA_sort3(input_file, bin_files_directory, h5_files_directory, kilosort_path, npy_matlab_path, path2cnfg, temporary_folder_directory)
    h5_files_directory = string(h5_files_directory);
    bin_files_directory = string(bin_files_directory);
    kilosort_path = string(kilosort_path);
    temporary_folder_directory = string(temporary_folder_directory);
    %%%%%%%%%%% split the filenames into parts that matter
    data = fileread(input_file);
    jsonfile = jsondecode(data);
    traces = jsonfile.traces;
    experiment_name = jsonfile.experiment_name;
    route_name = string(jsonfile.route_name);
    year = char(experiment_name);
    year = year(1:4);

    %%%%%%%%%%% declaring paths for data files and temporary folders
    addpath(genpath(kilosort_path)) % path to kilosort folder
    addpath(npy_matlab_path) % for converting to Phy
    disp(experiment_name)
    rootD                  = h5_files_directory +'/'+string(year)+'/'+string(experiment_name)+'/'; % path to the original h5 files
    rootZ                  = temporary_folder_directory; % the raw data binary file is in this folder
    rootH                  = temporary_folder_directory; % path to temporary binary file (same size as data, should be on fast SSD)
    rootO                  = bin_files_directory + '/'+string(experiment_name)+'/'; % path to the binary file
    disp(rootD)
    % Create the output folder, if it doesn't exist yet
    output_fname = string(experiment_name);
    if exist(rootZ+output_fname,'dir')
        fprintf('%s%s already exists.. No need to create new folder.. \n', rootZ, output_fname)
    else
        mkdir(rootZ+output_fname);
    end
    if exist(rootZ+output_fname+'/'+route_name,'dir')
        fprintf('%s%s/%s already exists.. No need to create new folder.. \n', rootZ, output_fname, route_name)
    else
        mkdir(rootZ+output_fname+'/'+route_name);
    end
    rootZ = rootZ+output_fname+'/'+route_name+'/'; % rename rootZ
 
    %%%%%%%%%%% run configuration file
    disp(traces)
    ops = kilosort_confs(traces, output_fname, path2cnfg, rootH, rootD);

    %%%%%%%%%%% save the binary file and load it
    binary_filename  = route_name+'.bin';
    fs               = rootZ+binary_filename;
    if exist(fs, 'dir')
        fprintf('%s is already there.. No need to copy.. \n', binary_filename)
    else
        concat_cmd = 'cat ';
        for i_idx = 1:length(traces)
            trace_name = char(traces{i_idx});
            if strcmp(string(trace_name(end-2:end)), '.h5')
                concat_cmd = concat_cmd+rootO+extractBefore(traces{i_idx},'.h5')+'.bin ';
            end
        end
        concat_cmd = concat_cmd+'> '+fs;
        disp(concat_cmd);
        system(concat_cmd);
    end
    ops.fbinary = fullfile(fs); 

    %%%%%%%%%%% Initilizing the GPU
    gpuDevice(1);

    %%%%%%%%%%% preprocess data to create temp_wh.dat
    fprintf('preprocess data to create temp_wh.dat \n')
    rez = preprocessDataSub(ops);
    rez = datashift2(rez, 1);
    
    %%%%%%%%%%% main tracking and template matching algorithm
    fprintf('main tracking and template matching algorithm \n')
    [rez, st3, tF] = extract_spikes(rez);
    rez = template_learning(rez, tF, st3);
    [rez, st3, tF] = trackAndSort(rez);

    %%%%%%%%%%% final clustering
    fprintf('final clustering \n')
    rez = final_clustering(rez, tF, st3);
    
    %%%%%%%%%%% final merges
    fprintf('final merges \n')
    rez = find_merges(rez, 1);

    %%%%%%%%%%% write to Phy
    fprintf('Saving results to Phy  \n')
    rezToPhy2(rez, rootZ);

    %%%%%%%%%%% save final results as rez2
    % if you want to save the results to a Matlab file...
    % discard features in final rez file (too slow to save)
    rez.cProj = [];
    rez.cProjPC = [];
    fprintf('Saving final results in rez2  \n')
    fname = fullfile(rootZ, 'rez2.mat');
    save(fname, 'rez', '-v7.3');
    concat_cmd = 'rm '+rootZ+binary_filename;
    disp(concat_cmd);
    system(concat_cmd);
