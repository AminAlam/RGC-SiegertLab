function MEA_sort(input_file)

clear fid
clear ans


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PATHS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
archive_path = "/home/malamalh/mounts/drives/archive-siegegrp";
fs3_path = "/home/malamalh/mounts/drives/fs3-siegegrp";
kilosort_path = "/home/malamalh/Documents/RGC/Kilosort";
npy_matlab_path = kilosort_path+"/npy-matlab";
pathToYourConfigFile = "/home/malamalh/Documents/RGC/configFiles"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% split the filenames into parts that matter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Opening %s \n', input_file)
fid = fopen(input_file);
data = textscan(fid,'%s');
fclose(fid);
data = string(data{:});
clear fid
clear ans

experiment_name    = string(data(1));
route_name         = string(data(2));
fprintf(experiment_name)

fname = {};
for i_idx = 1:length(data)
    trace_name = string(data(i_idx));
    trace_char = char(data(i_idx));
    if length(trace_char)>5
        if trace_char(1:5) == 'Trace'
            fname{end+1} = trace_name;
        end
    end
end

year               = char(experiment_name);
year               = year(1:4);

fprintf('Analysing %s of %s \n', string(route_name), string(experiment_name))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Declare all the folders
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath(kilosort_path)) % path to kilosort folder
addpath(npy_matlab_path) % for converting to Phy
rootD                  = fs3_path + '/Balint_RESTORED/Balint/MEA/'+string(year)+'/'+string(experiment_name)+'/'; % path to the original h5 files
rootZ                  = '/home/malamalh/Documents/RGC/data'; % the raw data binary file is in this folder
rootH                  = '/home/malamalh/Documents/RGC/data'; % path to temporary binary file (same size as data, should be on fast SSD)
rootO                  = archive_path + '/MEA_bin/'+string(experiment_name)+'/'; % path to the binary file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the output folder, if it doesn't exist yet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

rootZ                  = rootZ+output_fname+'/'+route_name+'/'; % rename rootZ

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run configuration file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ops = kilosort_confs(fname, output_fname, pathToYourConfigFile, rootH, rootD);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save the binary file and load it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

binary_filename  = route_name+'.bin';
fs               = rootZ+binary_filename;

if exist(fs)
    fprintf('%s is already there.. No need to copy.. \n', binary_filename)
else
    concat_cmd = 'cat ';
    for i_idx = 1:length(fname)
        trace_name = char(fname{i_idx})
        if string(trace_name(end-2:end)) == '.h5'
            concat_cmd = concat_cmd+rootO+extractBefore(fname{i_idx},'.h5')+'.bin ';
        end
    end
    concat_cmd = concat_cmd+'> '+fs;
    disp(concat_cmd);
    system(concat_cmd);
end

ops.fbinary = fullfile(fs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start the spike sorting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gpuDevice(1);

% this block runs all the steps of the algorithm
fprintf('Looking for data inside %s \n', rootZ)

% preprocess data to create temp_wh.dat
fprintf('preprocess data to create temp_wh.dat \n')
rez                = preprocessDataSub(ops);
rez                = datashift2(rez, 1);

% main tracking and template matching algorithm
fprintf('main tracking and template matching algorithm \n')
[rez, st3, tF]     = extract_spikes(rez);
rez                = template_learning(rez, tF, st3);
[rez, st3, tF]     = trackAndSort(rez);

% final clustering
fprintf('final clustering \n')
rez                = final_clustering(rez, tF, st3);

% final merges
fprintf('final merges \n')
rez                = find_merges(rez, 1);

% decide on cutoff
fprintf('decide on cutoff \n')
rez = set_cutoff(rez);

%fprintf('found %d good units \n', sum(rez.good>0))

% write to Phy
fprintf('Saving results to Phy  \n')
rezToPhy(rez, rootZ);

% if you want to save the results to a Matlab file...
% discard features in final rez file (too slow to save)
rez.cProj = [];
rez.cProjPC = [];

% save final results as rez2
fprintf('Saving final results in rez2  \n')
fname = fullfile(rootZ, 'rez2.mat');
save(fname, 'rez', '-v7.3');

concat_cmd = 'rm '+rootZ+binary_filename;
disp(concat_cmd);
system(concat_cmd);

close all;
