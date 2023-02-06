function ops = kilosort_confs(fname, output_fname, pathToYourConfigFile, rootH, rootD)
    
    fs                    = 20000; % sampling frequency
    ops.fs                        = fs;
    ops.trange                    = [0 Inf]; % time range to sort
    ops.criterionNoiseChannels    = 0.5; 
    
%     fprintf('Running configFile384.m \n')
    %%
    
    % frequency for high pass filtering (150)
    ops.fshigh = 300;   
    
    % threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
    ops.Th = [9 9];  
    
    % how important is the amplitude penalty (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot) 
    ops.lam = 20;  
    
    % splitting a cluster at the end requires at least this much isolation for each sub-cluster (max = 1)
    ops.AUCsplit = 0.8; 
    
    % minimum spike rate (Hz), if a cluster falls below this for too long it gets removed
    ops.minFR = 1/50; 
    
    % spatial constant in um for computing residual variance of spike
    ops.sigmaMask = 30; 
    
    % threshold crossings for pre-clustering (in PCA projection space)
    ops.ThPre = 8; 
    
    % spatial scale for datashift kernel
    ops.sig = 20;
    
    % type of data shifting (0 = none, 1 = rigid, 2 = nonrigid)
    ops.nblocks = 5;
    
    
    % danger, changing these settings can lead to fatal errors
    % options for determining PCs
    ops.spkTh           = -6;      % spike threshold in standard deviations (-6)
    ops.reorder         = 1;       % whether to reorder batches for drift correction. 
    ops.nskip           = 25;  % how many batches to skip for determining spike PCs
    
    ops.GPU                 = 1; % has to be 1, no CPU version yet, sorry
    % ops.Nfilt               = 1024; % max number of clusters
    ops.nfilt_factor        = 4; % max number of clusters per good channel (even temporary ones)
    ops.ntbuff              = 64;    % samples of symmetrical buffer for whitening and spike detection
    ops.NT                  = 64*1024+ ops.ntbuff; % must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory). 
    ops.whiteningRange      = 32; % number of channels to use for whitening each channel
    ops.nSkipCov            = 25; % compute whitening matrix from every N-th batch
    ops.scaleproc           = 200;   % int16 scaling of whitened data
    ops.nPCs                = 3; % how many PCs to project the spikes into
    ops.useRAM              = 0; % not yet available
    
    %%
    
    ops.fproc       = char(fullfile(rootH, 'temp_wh.dat')); % proc file on a fast SSD
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Create mini-patches and channel map file
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('Creating channel map file \n')
    fn                       = rootD+'/'+fname{1};
    disp(fn)
    mapping                  = h5read(fullfile(fn), '/mapping');
    
    % Create mini-patches
    radius = 200;
    valid_idx = mapping.x>=0;
    
    x                        = mapping.x;
    y                        = mapping.y;
    ch                       = mapping.channel(valid_idx);
    electrode                = mapping.electrode(valid_idx);
    
    [~, sorted]              = sort(electrode, 'ascend');
    xcoords                  = x(sorted);
    ycoords                  = y(sorted);
    
    Nchannels                = length(ch);
    connected                = true(Nchannels, 1);
    chanMap                  = 1:Nchannels;
    chanMap0ind              = chanMap - 1;
    
    kcoords                  = ones(length(mapping.x),1);
    
    all_used_channels = [];
    iteration = 1;
    while sum(valid_idx)>0
        % find upper left corner
        x = mapping.x(valid_idx);
        y = mapping.y(valid_idx);
        ch = mapping.channel(valid_idx);
    
        [~,b] = sort( sqrt( x.^2 + y.^2 ) );
        topPoint_x = x(b(1));
        topPoint_y = y(b(1));
    
        [d,b] = sort( sqrt( (topPoint_x-x).^2 + (topPoint_y-y).^2 ) );
        rng = d<radius;
        idx = b(rng);
        
        all_used_channels = [all_used_channels ; ch(idx)];
    
        for m_idx = 1:length(mapping.channel)
            if any( all_used_channels==mapping.channel(m_idx) )
                valid_idx(m_idx) = 0;
            end
            
            if any( ch(idx)==mapping.channel(m_idx) )
                kcoords(m_idx) = iteration;
            end
        end
        iteration=iteration + 1;
    end
    
    kcoords = kcoords(sorted);
    fprintf('There are %d clusters \n', length(unique(kcoords)))
    
    
    chanMapFile = output_fname+'_ChanMap.mat';
    save(fullfile(pathToYourConfigFile, chanMapFile), 'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')
    
    ops.chanMap     = char(fullfile(pathToYourConfigFile, chanMapFile));
    ops.NchanTOT    = Nchannels; % total number of channels in your recording





