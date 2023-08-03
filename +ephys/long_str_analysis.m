%% Load experiments and save in struct


animal = 'AP009';
use_workflow = {'lcr_passive'};
% use_workflow = {'lcr_passive_fullscreen'};
% use_workflow = {'lcr_passive','lcr_passive_fullscreen'};
% use_workflow = {'stim_wheel_right_stage1','stim_wheel_right_stage2'};
% use_workflow = 'sparse_noise';

recordings = ap.find_recordings(animal,use_workflow);

struct_ephys_data = struct;
missing_ephys = 0;
for use_rec=1:length(recordings)-2
    % use_rec = 1;
    % rec_day = '2023-06-28';
    % use_rec = strcmp(rec_day,{recordings.day});
    % use_rec = length(recordings)-1;

    rec_day = recordings(use_rec).day;
    rec_time = recordings(use_rec).protocol{end};

    verbose = true;

    ap.load_experiment

    ephys_path = dir(plab.locations.make_server_filename(animal,rec_day,[],'ephys'));
    if isempty(ephys_path)
        disp(['Day ' rec_day ' does not have ephys'])
        missing_ephys = missing_ephys  + 1; 
        continue
    end

    qMetrics_path = plab.locations.make_server_filename(animal,rec_day,[],'ephys','qMetrics');
    if isfolder(qMetrics_path)
        qMetricsExist = 0;
        qMetrics_probe_folders = dir(qMetrics_path);
        qMetrics_probe_folders = qMetrics_probe_folders(3:end);

        if  isempty(find([qMetrics_probe_folders.isdir]))
            % Load Julie's quality metrics
            qMetricsExist = ~isempty(dir(fullfile(qMetrics_path, 'qMetric*.mat'))) || ~isempty(dir(fullfile(qMetrics_path, 'templates._bc_qMetrics.parquet')));

            if qMetricsExist
                [param, qMetric] = bc_loadSavedMetrics(qMetrics_path);
                %             unitType = bc_getQualityUnitType(param, qMetric);
                load(fullfile(qMetrics_path, 'am_bc_unit_type.mat'))
                disp(['Loaded qMetrics']);

                % Define good units from labels
                bc_good_templates_idx_mask = am_bc_units == 1 | am_bc_units == 2;
                bc_good_templates_idx = find(bc_good_templates_idx_mask)';
                bc_good_templates_identity = bc_good_templates_idx - 1;

                bc_axons_idx_mask = am_bc_units == 3;
                bc_axons_idx = find(bc_axons_idx_mask)';
                bc_axons_identity = bc_axons_idx - 1;

                bc_noise_idx_mask = am_bc_units == 0;
                bc_noise_idx = find(bc_noise_idx_mask)';
                bc_noise_identity = bc_noise_idx - 1;

                bc_inc_axons_idx_mask = bc_good_templates_idx_mask | bc_axons_idx_mask;
                bc_inc_axons_idx = find(bc_inc_axons_idx_mask)';
                bc_inc_axons_identity = bc_inc_axons_idx - 1;

                disp(['Bombcell found ' num2str(length(bc_good_templates_idx)) ' good units']);
            end

        else
            qMetrics_probe_paths = strcat(qMetrics_path, filesep, qMetrics_probe_folders);

            for probe_idx=1:length(qMetrics_probe_paths)

                qMetrics_probe_path = qMetrics_probe_paths{probe_idx};

                % Load Julie's quality metrics
                qMetricsExist = ~isempty(dir(fullfile(qMetrics_probe_path, 'qMetric*.mat'))) || ~isempty(dir(fullfile(qMetrics_probe_path, 'templates._bc_qMetrics.parquet')));
                if qMetricsExist
                    [param, qMetric] = bc_loadSavedMetrics(qMetrics_probe_path);
                    %             unitType = bc_getQualityUnitType(param, qMetric);
                    load(fullfile(qMetrics_probe_path, 'am_bc_unit_type.mat'))
                    disp(['Loaded qMetrics']);

                    % Define good units from labels
                    bc_good_templates_idx_mask = am_bc_units == 1 | am_bc_units == 2;
                    bc_good_templates_idx = find(bc_good_templates_idx_mask)';
                    bc_good_templates_identity = bc_good_templates_idx - 1;

                    bc_axons_idx_mask = am_bc_units == 3;
                    bc_axons_idx = find(bc_axons_idx_mask)';
                    bc_axons_identity = bc_axons_idx - 1;

                    bc_noise_idx_mask = am_bc_units == 0;
                    bc_noise_idx = find(bc_noise_idx_mask)';
                    bc_noise_identity = bc_noise_idx - 1;

                    bc_inc_axons_idx_mask = bc_good_templates_idx_mask | bc_axons_idx_mask;
                    bc_inc_axons_idx = find(bc_inc_axons_idx_mask)';
                    bc_inc_axons_identity = bc_inc_axons_idx - 1;

                    disp(['Bombcell found ' num2str(length(bc_good_templates_idx)) ' good units']);

                end
            end
        end
    end


    % Throw out all non-good template data
    templates = templates(bc_good_templates_idx,:,:);
    template_depths = template_depths(bc_good_templates_idx);
    waveforms = waveforms(bc_good_templates_idx,:);
    templateDuration = templateDuration(bc_good_templates_idx);
    templateDuration_us = templateDuration_us(bc_good_templates_idx);

    % Throw out all non-good spike data
    good_spike_idx = ismember(spike_templates_0idx,bc_good_templates_identity);
    spike_times_openephys = spike_times_openephys(good_spike_idx);
    spike_templates_0idx = spike_templates_0idx(good_spike_idx);
    template_amplitudes = template_amplitudes(good_spike_idx);
    spike_depths = spike_depths(good_spike_idx);
    spike_times_timeline = spike_times_timeline(good_spike_idx);

    % Rename the spike templates according to the remaining templates
    % (and make 1-indexed from 0-indexed)
    new_spike_idx = nan(max(spike_templates_0idx)+1,1);
    new_spike_idx(bc_good_templates_idx) = 1:length(bc_good_templates_idx);
    spike_templates = new_spike_idx(spike_templates_0idx+1);


    %% plot

    trial_stim_values = vertcat(trial_events.values.TrialStimX);

%     AP_cellraster(stimOn_times,trial_stim_values);


    %% find no move trials

    % create matrix of times for stim onset
    timestep = 0.01;
    start_time = -0.5;
    end_time = 1;
    timevec = start_time:timestep:end_time;

    stim_frame = (-start_time)*(1/timestep)+1;

    time_stimulus = stimOn_times+timevec;

    % stim aligned wheel move
    t = timelite.timestamps;
    stim_wheel_move = interp1(t,+wheel_move,time_stimulus);
    no_move_trials = sum(stim_wheel_move(:,stim_frame:end),2)==0;

    %% Get striatum boundaries

    %%% Get correlation of MUA in sliding sindows
    depth_corr_window = 100; % MUA window in microns
    depth_corr_window_spacing = 50; % MUA window spacing in microns

    max_depths = 3840; % (hardcode, sometimes kilosort2 drops channels)

    depth_corr_bins = [0:depth_corr_window_spacing:(max_depths-depth_corr_window); ...
        (0:depth_corr_window_spacing:(max_depths-depth_corr_window))+depth_corr_window];
    depth_corr_bin_centers = depth_corr_bins(1,:) + diff(depth_corr_bins,[],1)/2;

    spike_binning_t = 0.01; % seconds
    spike_binning_t_edges = nanmin(spike_times_timeline):spike_binning_t:nanmax(spike_times_timeline);

    binned_spikes_depth = zeros(size(depth_corr_bins,2),length(spike_binning_t_edges)-1);
    for curr_depth = 1:size(depth_corr_bins,2)
        curr_depth_templates_idx = ...
            find(template_depths >= depth_corr_bins(1,curr_depth) & ...
            template_depths < depth_corr_bins(2,curr_depth));

        binned_spikes_depth(curr_depth,:) = histcounts(spike_times_timeline( ...
            ismember(spike_templates,curr_depth_templates_idx)),spike_binning_t_edges);
    end

    mua_corr = corrcoef(binned_spikes_depth');


    %%% Estimate start and end depths of striatum

    % % end of striatum: biggest (smoothed) drop in MUA correlation near end
    % groups_back = 30;
    % mua_corr_end = medfilt2(mua_corr(end-groups_back+1:end,end-groups_back+1:end),[3,3]);
    % mua_corr_end(triu(true(length(mua_corr_end)),0)) = nan;
    % median_corr = medfilt1(nanmedian(mua_corr_end,2),3);
    % [x,max_corr_drop] = min(diff(median_corr));
    % str_end = depth_corr_bin_centers(end-groups_back+max_corr_drop);

    % (new method)
    % end of striatum: minimum correlation on dim 1 * dim 2
    % (to look for the biggest dead space between correlated blocks)
    groups_back = 20;
    mua_corr_end = medfilt2(mua_corr(end-groups_back+1:end,end-groups_back+1:end),[3,3]);
    mua_corr_end(triu(true(length(mua_corr_end)),0)) = nan;
    mean_corr_dim1 = nanmean(mua_corr_end,2);
    mean_corr_dim2 = nanmean(mua_corr_end,1);
    mean_corr_mult = mean_corr_dim1.*mean_corr_dim2';
    [~,mean_corr_mult_min_idx] = min(mean_corr_mult);
    str_end = depth_corr_bin_centers(end-groups_back + mean_corr_mult_min_idx - 2); % err early: back up 2 (100 um)

    % start of striatum: look for ventricle
    % (by biggest gap between templates)
    min_gap = 200;
    sorted_template_depths = sort([0;template_depths]);
%     [max_gap,max_gap_idx] = max(diff(sorted_template_depths));
    [sort_gap, sort_gap_idx] = sort(diff(sorted_template_depths));

    if sort_gap(end) > min_gap
        str_start = sorted_template_depths(sort_gap_idx(end)+1)-1;
        if str_start > str_end && sort_gap(end-1) > min_gap
            str_start = sorted_template_depths(sort_gap_idx(end-1)+1)-1;
        end
    else
        str_start = sorted_template_depths(2);
    end

    str_depth = [str_start,str_end];


    %% psth
    %% - multiunit

    % create time vector around all stim onsets
    bin_window = 0.1;
    bin_edges = -0.5:bin_window:2;
    around_stim_time = stimOn_times + bin_edges;

    % possible stims
    possible_stim = unique(trial_stim_values);

    %% -- ?? striatum

    str_spikes = spike_depths>str_depth(1) & spike_depths<str_depth(2);
    str_spike_times = spike_times_timeline(str_spikes);

    % spike counts binned for each stim
    str_spikes_in_stim_time = nan(size(around_stim_time, 1)/length(possible_stim), size(around_stim_time, 2)-1, length(possible_stim));

    for stim_idx=1:length(possible_stim)
        this_stim = possible_stim(stim_idx);
        this_stim_time = around_stim_time(trial_stim_values == this_stim, :);
        % transpose to get the right shape
        str_spikes_in_stim_time(:, :, stim_idx) = cell2mat(arrayfun(@(trial_id) histcounts(str_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window;
    end
% 
%     figure;
%     plot(mean(str_spikes_in_stim_time(:, :, 1)))
%     hold on;
%     plot(mean(str_spikes_in_stim_time(:, :, 2)))
%     hold on;
%     plot(mean(str_spikes_in_stim_time(:, :, 3)))
%     title(['Striatum psth ' rec_day])

    %% unit responsivenes
    %% -- contra stim
    % sharp
    % create pre and post stim onsets
    bin_window_for_sharp = 0.1;
    pre_stim_time = stimOn_times - [bin_window_for_sharp 0];
    post_stim_time = stimOn_times + [0.05 bin_window_for_sharp+0.05];

    % possible stims
    possible_stim = unique(trial_stim_values);
    contra_stim = possible_stim(end);

    % get trials for this stim and no move
    good_trials = trial_stim_values == contra_stim & no_move_trials;

    unit_spikes_small_pre_stim = nan(length(bc_good_templates_idx), length(find(good_trials)));
    unit_spikes_small_post_stim = nan(length(bc_good_templates_idx), length(find(good_trials)));
    sharp_p_units = nan(length(bc_good_templates_idx), 1);
    for unit_idx=1:length(bc_good_templates_idx)

        unit_spikes = spike_templates == unit_idx & str_spikes;

        unit_spike_times = spike_times_timeline(unit_spikes);

        % spike counts binned pre/post stim
        this_stim_time = pre_stim_time(good_trials, :);
        unit_spikes_small_pre_stim(unit_idx, :)  = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_sharp;

        this_stim_time = post_stim_time(good_trials, :);
        unit_spikes_small_post_stim(unit_idx, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_sharp;

        % signed rank test
        sharp_p_units(unit_idx) = signrank(unit_spikes_small_post_stim(unit_idx, :), unit_spikes_small_pre_stim(unit_idx, :), 'tail', 'right');
    end

    sharp_responsive_units = find(sharp_p_units<0.05);
    sharp_unresponsive_units = find(sharp_p_units>0.05);

    % wide
    % create pre and post stim onsets
    bin_window_for_wide = 0.4;
    pre_stim_time = stimOn_times - [bin_window_for_wide 0];
    post_stim_time = stimOn_times + [0.05 bin_window_for_wide+0.05];

    % possible stims
    possible_stim = unique(trial_stim_values);
    contra_stim = possible_stim(end);

    % get trials for this stim and no move
    good_trials = trial_stim_values == contra_stim & no_move_trials;

    unit_spikes_small_pre_stim = nan(length(bc_good_templates_idx), length(find(good_trials)));
    unit_spikes_big_post_stim = nan(length(bc_good_templates_idx), length(find(good_trials)));
    wide_p_units = nan(length(bc_good_templates_idx), 1);
    for unit_idx=1:length(bc_good_templates_idx)

        unit_spikes = spike_templates == unit_idx & str_spikes;

        unit_spike_times = spike_times_timeline(unit_spikes);

        % spike counts binned pre/post stim
        this_stim_time = pre_stim_time(good_trials, :);
        unit_spikes_small_pre_stim(unit_idx, :)  = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_wide;

        this_stim_time = post_stim_time(good_trials, :);
        unit_spikes_big_post_stim(unit_idx, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_wide;

        % signed rank test
        wide_p_units(unit_idx) = signrank(unit_spikes_big_post_stim(unit_idx, :), unit_spikes_small_pre_stim(unit_idx, :), 'tail', 'right');
    end

    wide_responsive_units = find(wide_p_units<0.05);
    wide_unresponsive_units = find(wide_p_units>0.05);

    %% -- centre stim 
    % sharp
    % create pre and post stim onsets
    bin_window_for_sharp = 0.1;
    pre_stim_time = stimOn_times - [bin_window_for_sharp 0];
    post_stim_time = stimOn_times + [0.05 bin_window_for_sharp+0.05];

    % centre stim
    centre_stim = 0;

    % get trials for this stim and no move
    good_trials = trial_stim_values == centre_stim & no_move_trials;

    unit_spikes_small_pre_stim = nan(length(bc_good_templates_idx), length(find(good_trials)));
    unit_spikes_small_post_stim = nan(length(bc_good_templates_idx), length(find(good_trials)));
    centre_sharp_p_units = nan(length(bc_good_templates_idx), 1);
    for unit_idx=1:length(bc_good_templates_idx)

        unit_spikes = spike_templates == unit_idx & str_spikes;

        unit_spike_times = spike_times_timeline(unit_spikes);

        % spike counts binned pre/post stim
        this_stim_time = pre_stim_time(good_trials, :);
        unit_spikes_small_pre_stim(unit_idx, :)  = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_sharp;

        this_stim_time = post_stim_time(good_trials, :);
        unit_spikes_small_post_stim(unit_idx, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_sharp;

        % signed rank test
        centre_sharp_p_units(unit_idx) = signrank(unit_spikes_small_post_stim(unit_idx, :), unit_spikes_small_pre_stim(unit_idx, :), 'tail', 'right');
    end

    centre_sharp_responsive_units = find(centre_sharp_p_units<0.05);
    centre_sharp_unresponsive_units = find(centre_sharp_p_units>0.05);

    % wide
    % create pre and post stim onsets
    bin_window_for_wide = 0.4;
    pre_stim_time = stimOn_times - [bin_window_for_wide 0];
    post_stim_time = stimOn_times + [0.05 bin_window_for_wide+0.05];

    % centre stim
    centre_stim = 0;

    % get trials for this stim and no move
    good_trials = trial_stim_values == centre_stim & no_move_trials;

    unit_spikes_small_pre_stim = nan(length(bc_good_templates_idx), length(find(good_trials)));
    unit_spikes_big_post_stim = nan(length(bc_good_templates_idx), length(find(good_trials)));
    centre_wide_p_units = nan(length(bc_good_templates_idx), 1);
    for unit_idx=1:length(bc_good_templates_idx)

        unit_spikes = spike_templates == unit_idx & str_spikes;

        unit_spike_times = spike_times_timeline(unit_spikes);

        % spike counts binned pre/post stim
        this_stim_time = pre_stim_time(good_trials, :);
        unit_spikes_small_pre_stim(unit_idx, :)  = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_wide;

        this_stim_time = post_stim_time(good_trials, :);
        unit_spikes_big_post_stim(unit_idx, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_wide;

        % signed rank test
        centre_wide_p_units(unit_idx) = signrank(unit_spikes_big_post_stim(unit_idx, :), unit_spikes_small_pre_stim(unit_idx, :), 'tail', 'right');
    end

    centre_wide_responsive_units = find(centre_wide_p_units<0.05);
    centre_wide_unresponsive_units = find(centre_wide_p_units>0.05);

    %% raw data
    % spike counts binned for each stim

    % create time vector around all stim onsets
    bin_window = 0.001;
    bin_edges = -0.5:bin_window:2;
    around_stim_time = stimOn_times + bin_edges;

    % for plot
    bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;

    % contra stim
    % get trials for this stim and no move
    good_trials = trial_stim_values == contra_stim & no_move_trials;

    contra_all_spikes_in_stim_time = nan(length(bc_good_templates_idx), length(find(good_trials)), size(around_stim_time, 2)-1);

    for unit_idx=1:length(bc_good_templates_idx)

        unit_spikes = spike_templates == unit_idx & str_spikes;

        unit_spike_times = spike_times_timeline(unit_spikes);

        this_stim_time = around_stim_time(good_trials, :);

        contra_all_spikes_in_stim_time(unit_idx, :, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window;
    end

    contra_mean_all_spikes_in_stim_time = squeeze(mean(contra_all_spikes_in_stim_time, 2));

    % centre stim
    % get trials for this stim and no move
    good_trials = trial_stim_values == centre_stim & no_move_trials;

    centre_all_spikes_in_stim_time = nan(length(bc_good_templates_idx), length(find(good_trials)), size(around_stim_time, 2)-1);

    for unit_idx=1:length(bc_good_templates_idx)

        unit_spikes = spike_templates == unit_idx & str_spikes;

        unit_spike_times = spike_times_timeline(unit_spikes);

        this_stim_time = around_stim_time(good_trials, :);

        centre_all_spikes_in_stim_time(unit_idx, :, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
            1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window;
    end

    centre_mean_all_spikes_in_stim_time = squeeze(mean(centre_all_spikes_in_stim_time, 2));

    %% smooth
    % define gaussian window
    gauss_win = gausswin(51, 3)';

    % contra all units 
    contra_smooth_all_units = filter(gauss_win,sum(gauss_win),contra_mean_all_spikes_in_stim_time, [], 2);

    % centre all units
    centre_smooth_all_units = filter(gauss_win,sum(gauss_win),centre_mean_all_spikes_in_stim_time, [], 2);

    %% get baseline

%     contra_baseline = mean(mean(contra_all_spikes_in_stim_time(:,:,bin_centres>-0.2 & bin_centres<0), 3), 2);
%     centre_baseline = mean(mean(centre_all_spikes_in_stim_time(:,:,bin_centres>-0.2 & bin_centres<0), 3), 2);

    contra_smooth_baseline = mean(contra_smooth_all_units(:,bin_centres>-0.2 & bin_centres<0), 2);
    centre_smooth_baseline = mean(centre_smooth_all_units(:,bin_centres>-0.2 & bin_centres<0), 2);

    %% normalize

    % contra
    contra_norm_smooth_all_units = (contra_smooth_all_units - contra_smooth_baseline) ...
        ./ (contra_smooth_baseline + std(contra_smooth_baseline));

    % centre
    centre_norm_smooth_all_units = (centre_smooth_all_units - centre_smooth_baseline) ...
        ./ (centre_smooth_baseline + std(centre_smooth_baseline));

    %% save data in big struct
    struct_ephys_data(use_rec-missing_ephys).rec_day = rec_day;
    struct_ephys_data(use_rec-missing_ephys).bc_good_templates_idx = bc_good_templates_idx;

    struct_ephys_data(use_rec-missing_ephys).bin_edges = bin_edges;
    struct_ephys_data(use_rec-missing_ephys).bin_centres = bin_centres;

    struct_ephys_data(use_rec-missing_ephys).bin_window_for_sharp = bin_window_for_sharp;
    struct_ephys_data(use_rec-missing_ephys).bin_window_for_wide = bin_window_for_wide;

    struct_ephys_data(use_rec-missing_ephys).str_depth = str_depth;
    struct_ephys_data(use_rec-missing_ephys).str_spikes_in_stim_time = str_spikes_in_stim_time;

    struct_ephys_data(use_rec-missing_ephys).contra_mean_all_spikes_in_stim_time = contra_mean_all_spikes_in_stim_time;
    struct_ephys_data(use_rec-missing_ephys).contra_smooth_all_units = contra_smooth_all_units;
    struct_ephys_data(use_rec-missing_ephys).contra_smooth_baseline = contra_smooth_baseline;
    struct_ephys_data(use_rec-missing_ephys).contra_norm_smooth_all_units = contra_norm_smooth_all_units;

    struct_ephys_data(use_rec-missing_ephys).contra_sharp_responsive_units = sharp_responsive_units;
    struct_ephys_data(use_rec-missing_ephys).contra_sharp_unresponsive_units = sharp_unresponsive_units;
    struct_ephys_data(use_rec-missing_ephys).contra_wide_responsive_units = wide_responsive_units;
    struct_ephys_data(use_rec-missing_ephys).contra_wide_unresponsive_units = wide_unresponsive_units;

    struct_ephys_data(use_rec-missing_ephys).centre_mean_all_spikes_in_stim_time = centre_mean_all_spikes_in_stim_time;
    struct_ephys_data(use_rec-missing_ephys).centre_smooth_all_units = centre_smooth_all_units;
    struct_ephys_data(use_rec-missing_ephys).centre_smooth_baseline = centre_smooth_baseline;

    struct_ephys_data(use_rec-missing_ephys).centre_sharp_responsive_units = centre_sharp_responsive_units;
    struct_ephys_data(use_rec-missing_ephys).centre_sharp_unresponsive_units = centre_sharp_unresponsive_units;
    struct_ephys_data(use_rec-missing_ephys).centre_wide_responsive_units = centre_wide_responsive_units;
    struct_ephys_data(use_rec-missing_ephys).centre_wide_unresponsive_units = centre_wide_unresponsive_units;
    struct_ephys_data(use_rec-missing_ephys).centre_norm_smooth_all_units = centre_norm_smooth_all_units;

end

ephys_data = struct2table(struct_ephys_data);  %convert structure to table
% t_ephys_data = rmmissing(t_ephys_data);

save('AP009_str_ephys_data', "ephys_data");

clear all;

%% load and plot

save_fig_path = 'C:\Users\amarica\Documents\Lab stuff\Random figs\Long_str_stuff';

animal = 'AP009';
load('AP009_str_ephys_data');

for use_rec=1:height(ephys_data)

    rec_day = ephys_data.rec_day{use_rec};

    bin_centres = ephys_data.bin_centres(use_rec, :);

    contra_norm_smooth_all_units = ephys_data.contra_norm_smooth_all_units{use_rec};
    contra_sharp_responsive_units = ephys_data.contra_sharp_responsive_units{use_rec};
    contra_sharp_unresponsive_units = ephys_data.contra_sharp_unresponsive_units{use_rec};
    contra_wide_responsive_units = ephys_data.contra_wide_responsive_units{use_rec};
    contra_wide_unresponsive_units = ephys_data.contra_wide_unresponsive_units{use_rec};

    centre_norm_smooth_all_units = ephys_data.centre_norm_smooth_all_units{use_rec};
    centre_sharp_responsive_units = ephys_data.centre_sharp_responsive_units{use_rec};
    centre_sharp_unresponsive_units = ephys_data.centre_sharp_unresponsive_units{use_rec};
    centre_wide_responsive_units = ephys_data.centre_wide_responsive_units{use_rec};
    centre_wide_unresponsive_units = ephys_data.centre_wide_unresponsive_units{use_rec};

    % sorting units based on normalized values %%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % get spikes 50-150ms post stim
    post_stim_time = [0.05 0.15];
    post_bin_window = diff(post_stim_time);
    this_post_stim_time = bin_centres>post_stim_time(1) & bin_centres<post_stim_time(2);

    % contra
    contra_post_stim_spikes = mean(contra_norm_smooth_all_units(:, this_post_stim_time) / post_bin_window, 2);

    % sort for plotting
    [contra_sorted_post_stim_spikes, contra_sorted_units] = sort(contra_post_stim_spikes, 'descend');
    contra_sorted_norm_smooth_all_units = contra_norm_smooth_all_units(contra_sorted_units, :);

    % get sharp responsive sorted units for plotting
    [~, contra_sorted_sharp_responsive_units_idx] = sort(contra_post_stim_spikes(contra_sharp_responsive_units), 'descend');
    contra_sorted_sharp_responsive_units = contra_sharp_responsive_units(contra_sorted_sharp_responsive_units_idx);
    contra_sorted_norm_smooth_sharp_responsive_units = contra_norm_smooth_all_units(contra_sorted_sharp_responsive_units, :);

    % get sharp unresponsive sorted units for plotting
    [~, contra_sharp_unresponsive_units_idx] = sort(contra_post_stim_spikes(contra_sharp_unresponsive_units), 'descend');
    contra_sorted_sharp_unresponsive_units = contra_sharp_unresponsive_units(contra_sharp_unresponsive_units_idx);
    contra_sorted_norm_smooth_sharp_unresponsive_units = contra_norm_smooth_all_units(contra_sorted_sharp_unresponsive_units, :);

    % get wide responsive sorted units for plotting
    [~, contra_sorted_wide_responsive_units_idx] = sort(contra_post_stim_spikes(contra_wide_responsive_units), 'descend');
    contra_sorted_wide_responsive_units = contra_wide_responsive_units(contra_sorted_wide_responsive_units_idx);
    contra_sorted_norm_smooth_wide_responsive_units = contra_norm_smooth_all_units(contra_sorted_wide_responsive_units, :);

    % get wide unresponsive sorted units for plotting
    [~, contra_sorted_wide_unresponsive_units_idx] = sort(contra_post_stim_spikes(contra_wide_unresponsive_units), 'descend');
    contra_sorted_wide_unresponsive_units = contra_wide_unresponsive_units(contra_sorted_wide_unresponsive_units_idx);
    contra_sorted_norm_smooth_wide_unresponsive_units = contra_norm_smooth_all_units(contra_sorted_wide_unresponsive_units, :);

    % centre
    centre_post_stim_spikes = mean(centre_norm_smooth_all_units(:, this_post_stim_time) / post_bin_window, 2);

    % sort for plotting
    [centre_sorted_post_stim_spikes, centre_sorted_units] = sort(centre_post_stim_spikes, 'descend');
    centre_sorted_norm_smooth_all_units = centre_norm_smooth_all_units(centre_sorted_units, :);

    % get sharp responsive sorted units for plotting
    [~, centre_sorted_sharp_responsive_units_idx] = sort(centre_post_stim_spikes(centre_sharp_responsive_units), 'descend');
    centre_sorted_sharp_responsive_units = centre_sharp_responsive_units(centre_sorted_sharp_responsive_units_idx);
    centre_sorted_norm_smooth_sharp_responsive_units = centre_norm_smooth_all_units(centre_sorted_sharp_responsive_units, :);

    % get sharp unresponsive sorted units for plotting
    [~, centre_sharp_unresponsive_units_idx] = sort(centre_post_stim_spikes(centre_sharp_unresponsive_units), 'descend');
    centre_sorted_sharp_unresponsive_units = centre_sharp_unresponsive_units(centre_sharp_unresponsive_units_idx);
    centre_sorted_norm_smooth_sharp_unresponsive_units = centre_norm_smooth_all_units(centre_sorted_sharp_unresponsive_units, :);

    % get wide responsive sorted units for plotting
    [~, centre_sorted_wide_responsive_units_idx] = sort(centre_post_stim_spikes(centre_wide_responsive_units), 'descend');
    centre_sorted_wide_responsive_units = centre_wide_responsive_units(centre_sorted_wide_responsive_units_idx);
    centre_sorted_norm_smooth_wide_responsive_units = centre_norm_smooth_all_units(centre_sorted_wide_responsive_units, :);

    % get wide unresponsive sorted units for plotting
    [~, centre_sorted_wide_unresponsive_units_idx] = sort(centre_post_stim_spikes(centre_wide_unresponsive_units), 'descend');
    centre_sorted_wide_unresponsive_units = centre_wide_unresponsive_units(centre_sorted_wide_unresponsive_units_idx);
    centre_sorted_norm_smooth_wide_unresponsive_units = centre_norm_smooth_all_units(centre_sorted_wide_unresponsive_units, :);

    % plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % - contra %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    contra_stim_fig = figure('Position', get(0, 'Screensize'));
    sgtitle(['Contra responses ' rec_day]);
    
    % -- sharp 
    if ~isempty(max(abs(contra_sorted_norm_smooth_sharp_responsive_units), [],"all"))
        upper_caxis = max(max(abs(contra_sorted_norm_smooth_sharp_responsive_units), [],"all"), ...
            max(abs(contra_sorted_norm_smooth_sharp_unresponsive_units), [],"all"));
    else
        upper_caxis = max(abs(contra_sorted_norm_smooth_sharp_unresponsive_units), [],"all");
    end

    % responsive cells
    resp = subplot(2,3,1);
    imagesc(bin_centres, [], contra_sorted_norm_smooth_sharp_responsive_units);
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    colormap(resp, AP_colormap('BWR', [], 0.7));
    caxis([-upper_caxis upper_caxis]);
    colorbar;
    title('Responsive cells');
    ylabel('Sharp', 'FontWeight', 'bold', 'FontSize', 14);


    % unresponsive cells
    unresp = subplot(2,3,2);
    imagesc(bin_centres, [], contra_sorted_norm_smooth_sharp_unresponsive_units);
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    colormap(unresp, AP_colormap('BWR', [], 0.7));
    caxis([-upper_caxis upper_caxis]);
    colorbar;
    title('Unresponsive cells');

    % mean traces
    subplot(2,3,3);
    plot(bin_centres, mean(contra_sorted_norm_smooth_sharp_responsive_units, 1));
    hold on;
    plot(bin_centres, mean(contra_sorted_norm_smooth_sharp_unresponsive_units, 1));
    hold on;
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    legend({'Responsive', 'Unresponsive'});


    % -- wide
    if ~isempty(max(abs(contra_sorted_norm_smooth_wide_responsive_units), [],"all"))
        upper_caxis = max(max(abs(contra_sorted_norm_smooth_wide_responsive_units), [],"all"), ...
            max(abs(contra_sorted_norm_smooth_wide_unresponsive_units), [],"all"));
    else
        upper_caxis = max(abs(contra_sorted_norm_smooth_wide_unresponsive_units), [],"all");
    end

    % responsive cells
    resp = subplot(2,3,4);
    imagesc(bin_centres, [], contra_sorted_norm_smooth_wide_responsive_units);
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    colormap(resp, AP_colormap('BWR', [], 0.7));
    caxis([-upper_caxis upper_caxis]);
    colorbar;
    title('Responsive cells');
    ylabel('Wide', 'FontWeight', 'bold', 'FontSize', 14);


    % unresponsive cells
    unresp = subplot(2,3,5);
    imagesc(bin_centres, [], contra_sorted_norm_smooth_wide_unresponsive_units);
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    colormap(unresp, AP_colormap('BWR', [], 0.7));
    caxis([-upper_caxis upper_caxis]);
    colorbar;
    title('Unresponsive cells');

    % mean traces
    subplot(2,3,6);
    plot(bin_centres, mean(contra_sorted_norm_smooth_wide_responsive_units, 1));
    hold on;
    plot(bin_centres, mean(contra_sorted_norm_smooth_wide_unresponsive_units, 1));
    hold on;
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    legend({'Responsive', 'Unresponsive'});

    % save this fig
    contra_stim_fig_name = [animal '_' rec_day '_Contra.tif'];
    contra_stim_fig_path = fullfile(save_fig_path, contra_stim_fig_name);
    saveas(contra_stim_fig, contra_stim_fig_path);


    % - centre %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    centre_stim_fig = figure('Position', get(0, 'Screensize'));
    sgtitle(['Centre responses ' rec_day]);
    
    % -- sharp
    if ~isempty(max(abs(centre_sorted_norm_smooth_sharp_responsive_units), [],"all"))
        upper_caxis = max(max(abs(centre_sorted_norm_smooth_sharp_responsive_units), [],"all"), ...
            max(abs(centre_sorted_norm_smooth_sharp_unresponsive_units), [],"all"));
    else
        upper_caxis = max(abs(centre_sorted_norm_smooth_sharp_unresponsive_units), [],"all");
    end

    % responsive cells
    resp = subplot(2,3,1);
    imagesc(bin_centres, [], centre_sorted_norm_smooth_sharp_responsive_units);
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    colormap(resp, AP_colormap('BWR', [], 0.7));
    caxis([-upper_caxis upper_caxis]);
    colorbar;
    title('Responsive cells');
    ylabel('Sharp', 'FontWeight', 'bold', 'FontSize', 14);


    % unresponsive cells
    unresp = subplot(2,3,2);
    imagesc(bin_centres, [], centre_sorted_norm_smooth_sharp_unresponsive_units);
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    colormap(unresp, AP_colormap('BWR', [], 0.7));
    caxis([-upper_caxis upper_caxis]);
    colorbar;
    title('Unresponsive cells');

    % mean traces
    subplot(2,3,3);
    plot(bin_centres, mean(centre_sorted_norm_smooth_sharp_responsive_units, 1));
    hold on;
    plot(bin_centres, mean(centre_sorted_norm_smooth_sharp_unresponsive_units, 1));
    hold on;
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    legend({'Responsive', 'Unresponsive'});


    % -- wide
    if ~isempty(max(abs(centre_sorted_norm_smooth_wide_responsive_units), [],"all"))
        upper_caxis = max(max(abs(centre_sorted_norm_smooth_wide_responsive_units), [],"all"), ...
            max(abs(centre_sorted_norm_smooth_wide_unresponsive_units), [],"all"));
    else
        upper_caxis = max(abs(centre_sorted_norm_smooth_wide_unresponsive_units), [],"all");
    end

    % responsive cells
    resp = subplot(2,3,4);
    imagesc(bin_centres, [], centre_sorted_norm_smooth_wide_responsive_units);
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    colormap(resp, AP_colormap('BWR', [], 0.7));
    caxis([-upper_caxis upper_caxis]);
    colorbar;
    title('Responsive cells');
    ylabel('Wide', 'FontWeight', 'bold', 'FontSize', 14);


    % unresponsive cells
    unresp = subplot(2,3,5);
    imagesc(bin_centres, [], centre_sorted_norm_smooth_wide_unresponsive_units);
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    colormap(unresp, AP_colormap('BWR', [], 0.7));
    caxis([-upper_caxis upper_caxis]);
    colorbar;
    title('Unresponsive cells');

    % mean traces
    subplot(2,3,6);
    plot(bin_centres, mean(centre_sorted_norm_smooth_wide_responsive_units, 1));
    hold on;
    plot(bin_centres, mean(centre_sorted_norm_smooth_wide_unresponsive_units, 1));
    hold on;
    xline(0, 'LineWidth', 1);
    xline(0.5, 'LineWidth', 1);
    legend({'Responsive', 'Unresponsive'});
    

    % save this fig
    centre_stim_fig_name = [animal '_' rec_day '_Centre.tif'];
    centre_stim_fig_path = fullfile(save_fig_path, centre_stim_fig_name);
    saveas(centre_stim_fig, centre_stim_fig_path);

end

%% get responsive units and plot number across days
% contra
all_contra_sharp_responsive_units = {ephys_data.contra_sharp_responsive_units{:}};
all_contra_wide_responsive_units = {ephys_data.contra_wide_responsive_units{:}};
num_cells_contra_sharp_responsive_units = cellfun(@(x) length(x), all_contra_sharp_responsive_units);
num_cells_contra_wide_responsive_units = cellfun(@(x) length(x), all_contra_wide_responsive_units);

% centre
all_centre_sharp_responsive_units = {ephys_data.centre_sharp_responsive_units{:}};
all_centre_wide_responsive_units = {ephys_data.centre_wide_responsive_units{:}};
num_cells_centre_sharp_responsive_units = cellfun(@(x) length(x), all_centre_sharp_responsive_units);
num_cells_centre_wide_responsive_units = cellfun(@(x) length(x), all_centre_wide_responsive_units);

% plots
contra_num_cells_fig = figure;
plot(num_cells_contra_sharp_responsive_units, '-o')
hold on;
plot(num_cells_contra_wide_responsive_units, '-o')
legend({'Sharp', 'Wide'})
title('Contra stim. Number of responsive cells across days')
contra_stim_fig_name = [animal '_Contra_stim_Number_responsive_cells.tif'];
contra_stim_fig_path = fullfile(save_fig_path, contra_stim_fig_name);
saveas(contra_num_cells_fig, contra_stim_fig_path);

centre_num_cells_fig = figure;
plot(num_cells_centre_sharp_responsive_units, '-o')
hold on;
plot(num_cells_centre_wide_responsive_units, '-o')
legend({'Sharp', 'Wide'})
title('Centre stim. Number of responsive cells across days')
centre_stim_fig_name = [animal '_centre_stim_Number_responsive_cells.tif'];
centre_stim_fig_path = fullfile(save_fig_path, centre_stim_fig_name);
saveas(centre_num_cells_fig, centre_stim_fig_path);

%% get max amplitudes across days
all_contra_sharp_mean_max_amplitude = nan(height(ephys_data), 1);
all_contra_wide_mean_max_amplitude = nan(height(ephys_data), 1);
all_centre_sharp_mean_max_amplitude = nan(height(ephys_data), 1);
all_centre_wide_mean_max_amplitude = nan(height(ephys_data), 1);
for use_rec=1:height(ephys_data)
    
    rec_day = ephys_data.rec_day{use_rec};

    bin_centres = ephys_data.bin_centres(use_rec, :);

    % contra
    contra_norm_smooth_all_units = ephys_data.contra_norm_smooth_all_units{use_rec};
    contra_sharp_responsive_units = ephys_data.contra_sharp_responsive_units{use_rec};
    contra_wide_responsive_units = ephys_data.contra_wide_responsive_units{use_rec};

    contra_sharp_mean_max_amplitude = max(mean(contra_norm_smooth_all_units(contra_sharp_responsive_units, :), 1));
    contra_wide_mean_max_amplitude = max(mean(contra_norm_smooth_all_units(contra_wide_responsive_units, :), 1));
    all_contra_sharp_mean_max_amplitude(use_rec) = contra_sharp_mean_max_amplitude;
    all_contra_wide_mean_max_amplitude(use_rec) = contra_wide_mean_max_amplitude;
    
    % centre
    centre_norm_smooth_all_units = ephys_data.centre_norm_smooth_all_units{use_rec};contra
    centre_sharp_responsive_units = ephys_data.centre_sharp_responsive_units{use_rec};
    centre_wide_responsive_units = ephys_data.centre_wide_responsive_units{use_rec};  

    centre_sharp_mean_max_amplitude = max(mean(centre_norm_smooth_all_units(centre_sharp_responsive_units, :), 1));
    centre_wide_mean_max_amplitude = max(mean(centre_norm_smooth_all_units(centre_wide_responsive_units, :), 1));
    all_centre_sharp_mean_max_amplitude(use_rec) = centre_sharp_mean_max_amplitude;
    all_centre_wide_mean_max_amplitude(use_rec) = centre_wide_mean_max_amplitude;
end

contra_max_ampl_fig = figure;
plot(all_contra_sharp_mean_max_amplitude, '-o')
hold on;
plot(all_contra_wide_mean_max_amplitude, '-o')
legend({'Sharp', 'Wide'})
title('Contra Max amplitudes')
contra_stim_fig_name = [animal '_contra_stim_Max_amplitudes.tif'];
contra_stim_fig_path = fullfile(save_fig_path, contra_stim_fig_name);
saveas(contra_max_ampl_fig, contra_stim_fig_path);

centre_max_ampl_fig = figure;
plot(all_centre_sharp_mean_max_amplitude, '-o')
hold on;
plot(all_centre_wide_mean_max_amplitude, '-o')
legend({'Sharp', 'Wide'})
title('Centre Max amplitudes')
centre_stim_fig_name = [animal '_centre_stim_Max_amplitudes.tif'];
centre_stim_fig_path = fullfile(save_fig_path, centre_stim_fig_name);
saveas(centre_max_ampl_fig, centre_stim_fig_path);