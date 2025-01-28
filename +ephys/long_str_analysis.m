%% Load experiments and save in struct

% load('swr_bhv_data.mat');



% save_fig_path = 'C:\Users\amarica\Documents\Lab stuff\Random figs\Long_str_stuff\Passive';
% animals = swr_bhv_data.animals;

% animals = {'AM019'};

% animals = {'AM008', 'AP008', 'AP009', 'AM011', 'AM012', ...
%    'AM014', 'AM015', 'AM016', 'AM017'}

animals = {'AP023'}; %, 'AM027'}; %, 'AM022'};

for animal_idx=1:length(animals)
    animal = animals{animal_idx};
    workflow = {'lcr_passive'};
    recordings = plab.find_recordings(animal, [], workflow);

    struct_ephys_data = struct;
    missing_ephys = 0;

    % for AM021 from 05/04 - end (day 6 - end)
    for use_rec=6:length(recordings)
        % use_rec = 1;
        %     rec_day = '2023-07-05';
        %     use_rec = strcmp(rec_day,{recordings.day});
        % use_rec = length(recordings)-1;

        rec_day = recordings(use_rec).day;
        rec_time = recordings(use_rec).recording{end};

        if strcmp(animal, 'AP008') && strcmp(rec_day, '2023-07-11')
            continue
        end

        if strcmp(animal, 'AP009') && (strcmp(rec_day, '2023-07-14') || strcmp(rec_day, '2023-07-12'))
            continue
        end

        verbose = true;

        load_parts.behavior = true;
        load_parts.ephys = true;
        

        ap.load_recording

        ephys_path = dir(plab.locations.filename('server', animal,rec_day,[],'ephys'));
        if isempty(ephys_path)
            disp(['Day ' rec_day ' does not have ephys'])
            missing_ephys = missing_ephys  + 1;
            continue
        end

        %% plot

        trial_stim_values = vertcat(trial_events.values.TrialStimX);
        trial_stim_values = trial_stim_values(1:length(stimOn_times));


%         %         ap.cellraster(stimOn_times,trial_stim_values);
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

%          ap.cellraster(stimOn_times(no_move_trials),trial_stim_values(no_move_trials));
 

        good_templates = logical(ones(1, size(templates, 1)));
        good_templates_idx = 1:size(templates, 1);


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

        [sorted_template_depths, sorted_template_depth_idx] = sort(template_depths);

        % find str start where unit depth distr not linear anymore
        idx_str_start = ischange(sorted_template_depths, 'linear','MaxNumChanges',1);

        %         test_x_axis = 1:length(sorted_template_depths);
        %         figure;
        %         plot(sorted_template_depths, 'o')
        %         hold on
        %         plot(test_x_axis(test_a), sorted_template_depths(test_a), '*')
        %         title(rec_day)


        str_start = sorted_template_depths(idx_str_start);
        str_end = sorted_template_depths(end);
        str_depth = [str_start,str_end];

        %%% Get correlation of MUA in sliding sindows
        %         depth_corr_window = 100; % MUA window in microns
        %         depth_corr_window_spacing = 50; % MUA window spacing in microns
        %
        %         max_depths = 3840; % (hardcode, sometimes kilosort2 drops channels)
        %
        %         depth_corr_bins = [0:depth_corr_window_spacing:(max_depths-depth_corr_window); ...
        %             (0:depth_corr_window_spacing:(max_depths-depth_corr_window))+depth_corr_window];
        %         depth_corr_bin_centers = depth_corr_bins(1,:) + diff(depth_corr_bins,[],1)/2;
        %
        %         spike_binning_t = 0.01; % seconds
        %         spike_binning_t_edges = nanmin(spike_times_timeline):spike_binning_t:nanmax(spike_times_timeline);
        %
        %         binned_spikes_depth = zeros(size(depth_corr_bins,2),length(spike_binning_t_edges)-1);
        %         for curr_depth = 1:size(depth_corr_bins,2)
        %             curr_depth_templates_idx = ...
        %                 find(template_depths >= depth_corr_bins(1,curr_depth) & ...
        %                 template_depths < depth_corr_bins(2,curr_depth));
        %
        %             binned_spikes_depth(curr_depth,:) = histcounts(spike_times_timeline( ...
        %                 ismember(spike_templates,curr_depth_templates_idx)),spike_binning_t_edges);
        %         end
        %
        %         mua_corr = corrcoef(binned_spikes_depth');
        %
        %
        %         %%% Estimate start and end depths of striatum
        %
        %         % % end of striatum: biggest (smoothed) drop in MUA correlation near end
        %         % groups_back = 30;
        %         % mua_corr_end = medfilt2(mua_corr(end-groups_back+1:end,end-groups_back+1:end),[3,3]);
        %         % mua_corr_end(triu(true(length(mua_corr_end)),0)) = nan;
        %         % median_corr = medfilt1(nanmedian(mua_corr_end,2),3);
        %         % [x,max_corr_drop] = min(diff(median_corr));
        %         % str_end = depth_corr_bin_centers(end-groups_back+max_corr_drop);
        %
        %         % (new method)
        %         % end of striatum: minimum correlation on dim 1 * dim 2
        %         % (to look for the biggest dead space between correlated blocks)
        %         groups_back = 20;
        %         mua_corr_end = medfilt2(mua_corr(end-groups_back+1:end,end-groups_back+1:end),[3,3]);
        %         mua_corr_end(triu(true(length(mua_corr_end)),0)) = nan;
        %         mean_corr_dim1 = nanmean(mua_corr_end,2);
        %         mean_corr_dim2 = nanmean(mua_corr_end,1);
        %         mean_corr_mult = mean_corr_dim1.*mean_corr_dim2';
        %         [~,mean_corr_mult_min_idx] = min(mean_corr_mult);
        %         str_end = depth_corr_bin_centers(end-groups_back + mean_corr_mult_min_idx - 2); % err early: back up 2 (100 um)
        %
        %         % start of striatum: look for ventricle
        %         % (by biggest gap between templates)
        %         min_gap = 200;
        %         sorted_good_template_depths = sort([0;template_depths(good_templates)]);
        %         %     [max_gap,max_gap_idx] = max(diff(sorted_template_depths));
        %         [sort_gap, sort_gap_idx] = sort(diff(sorted_good_template_depths));
        %
        %         if sort_gap(end) > min_gap
        %             str_start = sorted_good_template_depths(sort_gap_idx(end)+1)-1;
        %             if str_start > str_end && sort_gap(end-1) > min_gap
        %                 str_start = sorted_good_template_depths(sort_gap_idx(end-1)+1)-1;
        %             end
        %         else
        %             str_start = sorted_good_template_depths(2);
        %         end
        %
        %         str_depth = [str_start,str_end];


        %% psth
        %% - multiunit

        % create time vector around all stim onsets
        bin_window = 0.1;
        bin_edges = -0.5:bin_window:2;
        around_stim_time = stimOn_times + bin_edges;

        % possible stims
        possible_stim = unique(trial_stim_values, 'sorted');

        %% -- ?? striatum

        str_spikes = spike_depths>str_depth(1) & spike_depths<str_depth(2);
        str_spike_times = spike_times_timeline(str_spikes);

        % spike counts binned for each stim
        %     str_spikes_in_stim_time = nan(length(stimOn_times)/length(possible_stim), size(around_stim_time, 2)-1, length(possible_stim));
        %
        %     for stim_idx=1:length(possible_stim)
        %         this_stim = possible_stim(stim_idx);
        %         this_stim_time = around_stim_time(trial_stim_values == this_stim, :);
        %         % transpose to get the right shape
        %         str_spikes_in_stim_time(:, :, stim_idx) = cell2mat(arrayfun(@(trial_id) histcounts(str_spike_times, this_stim_time(trial_id,:))', ...
        %             1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window;
        %     end

        str_spikes_in_stim_time = cell2mat(arrayfun(@(trial_id) histcounts(str_spike_times, around_stim_time(trial_id,:))', ...
            1:size(around_stim_time, 1), 'UniformOutput',false))' / bin_window;

        %     test_a = grpstats(str_spikes_in_stim_time, trial_stim_values, "numel");
        test_b = grpstats(str_spikes_in_stim_time, trial_stim_values, "mean");

        possible_stim_for_legend = arrayfun(@num2str, possible_stim, 'UniformOutput', 0);

        figure;
        plot(test_b')
        legend(possible_stim_for_legend)
        title(['Striatum psth ' rec_day])


        %% unit responsivenes
        %% -- contra stim
        % sharp
        % create pre and post stim onsets
        bin_window_for_sharp = 0.1;
        pre_stim_time = stimOn_times - [bin_window_for_sharp 0];
        post_stim_time = stimOn_times + [0.05 bin_window_for_sharp+0.05];

        % contra stim
        contra_stim = 90;

        % get trials for this stim and no move
        contra_good_trials = (trial_stim_values == contra_stim) & no_move_trials;

        unit_spikes_small_pre_stim = nan(length(good_templates_idx), length(find(contra_good_trials)));
        unit_spikes_small_post_stim = nan(length(good_templates_idx), length(find(contra_good_trials)));
        contra_sharp_p_units = nan(length(good_templates_idx), 1);
        for unit_idx=1:length(good_templates_idx)

            unit_spikes = spike_templates == good_templates_idx(unit_idx) & str_spikes;

            unit_spike_times = spike_times_timeline(unit_spikes);

            % spike counts binned pre/post stim
            this_stim_time = pre_stim_time(contra_good_trials, :);
            unit_spikes_small_pre_stim(unit_idx, :)  = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
                1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_sharp;

            this_stim_time = post_stim_time(contra_good_trials, :);
            unit_spikes_small_post_stim(unit_idx, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
                1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_sharp;

            % signed rank test
            contra_sharp_p_units(unit_idx) = signrank(unit_spikes_small_post_stim(unit_idx, :), unit_spikes_small_pre_stim(unit_idx, :), 'tail', 'right');
        end

        % wide
        % create pre and post stim onsets
        bin_window_for_wide = 0.4;
        pre_stim_time = stimOn_times - [bin_window_for_wide 0];
        post_stim_time = stimOn_times + [0.05 bin_window_for_wide+0.05];

        unit_spikes_small_pre_stim = nan(length(good_templates_idx), length(find(contra_good_trials)));
        unit_spikes_big_post_stim = nan(length(good_templates_idx), length(find(contra_good_trials)));
        contra_wide_p_units = nan(length(good_templates_idx), 1);
        for unit_idx=1:length(good_templates_idx)

            unit_spikes = spike_templates == good_templates_idx(unit_idx) & str_spikes;

            unit_spike_times = spike_times_timeline(unit_spikes);

            % spike counts binned pre/post stim
            this_stim_time = pre_stim_time(contra_good_trials, :);
            unit_spikes_small_pre_stim(unit_idx, :)  = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
                1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_wide;

            this_stim_time = post_stim_time(contra_good_trials, :);
            unit_spikes_big_post_stim(unit_idx, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
                1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_wide;

            % signed rank test
            contra_wide_p_units(unit_idx) = signrank(unit_spikes_big_post_stim(unit_idx, :), unit_spikes_small_pre_stim(unit_idx, :), 'tail', 'right');
        end

        %% -- centre stim
        % sharp
        % create pre and post stim onsets
        bin_window_for_sharp = 0.1;
        pre_stim_time = stimOn_times - [bin_window_for_sharp 0];
        post_stim_time = stimOn_times + [0.05 bin_window_for_sharp+0.05];

        % centre stim
        centre_stim = 0;

        % get trials for this stim and no move
        centre_good_trials = trial_stim_values == centre_stim & no_move_trials;

        unit_spikes_small_pre_stim = nan(length(good_templates_idx), length(find(centre_good_trials)));
        unit_spikes_small_post_stim = nan(length(good_templates_idx), length(find(centre_good_trials)));
        centre_sharp_p_units = nan(length(good_templates_idx), 1);
        for unit_idx=1:length(good_templates_idx)

            unit_spikes = spike_templates == good_templates_idx(unit_idx) & str_spikes;

            unit_spike_times = spike_times_timeline(unit_spikes);

            % spike counts binned pre/post stim
            this_stim_time = pre_stim_time(centre_good_trials, :);
            unit_spikes_small_pre_stim(unit_idx, :)  = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
                1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_sharp;

            this_stim_time = post_stim_time(centre_good_trials, :);
            unit_spikes_small_post_stim(unit_idx, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
                1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_sharp;

            % signed rank test
            centre_sharp_p_units(unit_idx) = signrank(unit_spikes_small_post_stim(unit_idx, :), unit_spikes_small_pre_stim(unit_idx, :), 'tail', 'right');
        end

        % wide
        % create pre and post stim onsets
        bin_window_for_wide = 0.4;
        pre_stim_time = stimOn_times - [bin_window_for_wide 0];
        post_stim_time = stimOn_times + [0.05 bin_window_for_wide+0.05];

        % centre stim
        centre_stim = 0;
        unit_spikes_small_pre_stim = nan(length(good_templates_idx), length(find(centre_good_trials)));
        unit_spikes_big_post_stim = nan(length(good_templates_idx), length(find(centre_good_trials)));
        centre_wide_p_units = nan(length(good_templates_idx), 1);
        for unit_idx=1:length(good_templates_idx)

            unit_spikes = spike_templates == good_templates_idx(unit_idx) & str_spikes;

            unit_spike_times = spike_times_timeline(unit_spikes);

            % spike counts binned pre/post stim
            this_stim_time = pre_stim_time(centre_good_trials, :);
            unit_spikes_small_pre_stim(unit_idx, :)  = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
                1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_wide;

            this_stim_time = post_stim_time(centre_good_trials, :);
            unit_spikes_big_post_stim(unit_idx, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
                1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window_for_wide;

            % signed rank test
            centre_wide_p_units(unit_idx) = signrank(unit_spikes_big_post_stim(unit_idx, :), unit_spikes_small_pre_stim(unit_idx, :), 'tail', 'right');
        end

        %% raw data
        % spike counts binned for each stim

        % create time vector around all stim onsets
        bin_window = 0.001;
        bin_edges = -0.5:bin_window:2;
        around_stim_time = stimOn_times + bin_edges;

        % for plot
        bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;

        % contra stim
        contra_all_spikes_in_stim_time = nan(length(good_templates_idx), length(find(contra_good_trials)), size(around_stim_time, 2)-1);

        for unit_idx=1:length(good_templates_idx)

            unit_spikes = spike_templates == good_templates_idx(unit_idx) & str_spikes;

            unit_spike_times = spike_times_timeline(unit_spikes);

            this_stim_time = around_stim_time(contra_good_trials, :);

            contra_all_spikes_in_stim_time(unit_idx, :, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
                1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window;
        end

        % centre stim
        % get trials for this stim and no move
        centre_all_spikes_in_stim_time = nan(length(good_templates_idx), length(find(centre_good_trials)), size(around_stim_time, 2)-1);

        for unit_idx=1:length(good_templates_idx)

            unit_spikes = spike_templates == good_templates_idx(unit_idx) & str_spikes;

            unit_spike_times = spike_times_timeline(unit_spikes);

            this_stim_time = around_stim_time(centre_good_trials, :);

            centre_all_spikes_in_stim_time(unit_idx, :, :) = cell2mat(arrayfun(@(trial_id) histcounts(unit_spike_times, this_stim_time(trial_id,:))', ...
                1:size(this_stim_time, 1), 'UniformOutput',false))' / bin_window;
        end


        %% classify str neuron type - needs fixing

        % Split striatal/nonstriatal cells
        str_good_templates = template_depths(good_templates) >= str_depth(1) & template_depths(good_templates) <= str_depth(2);
        %     non_str_templates = ~str_templates;

        % Define the window to look for spiking statistics in (spikes go in and
        % out, so take the bin with the largest firing rate for each cell and work
        % with that one)
        % spiking_stat_window = 60*5; % seconds
        % spiking_stat_bins = min(spike_times_timeline):spiking_stat_window: ...
        %     max(spike_times_timeline);

        % % (for whole session)
        spiking_stat_window = max(spike_times_timeline)-min(spike_times_timeline);
        spiking_stat_bins = [min(spike_times_timeline),max(spike_times_timeline)];

        % Get firing rate across the session
        bin_spikes = nan(size(templates(good_templates),1), ...
            length(spiking_stat_bins)-1);
        for curr_template_idx = 1:length(good_templates_idx)
            bin_spikes(curr_template_idx,:) = ...
                histcounts(spike_times_timeline(spike_templates == curr_template_idx), ...
                spiking_stat_bins);
        end
        min_spikes = 10;
        use_spiking_stat_bins = bsxfun(@ge,bin_spikes,prctile(bin_spikes,80,2)) & bin_spikes > min_spikes;
        spike_rate = sum(bin_spikes.*use_spiking_stat_bins,2)./ ...
            (sum(use_spiking_stat_bins,2)*spiking_stat_window);

        % Get proportion of ISI > 2s (Yamin/Cohen 2013) and CV2 (Stalnaker/Schoenbaum 2016)
        prop_long_isi = nan(size(templates(good_templates),1),1);
        cv2 = nan(size(templates(good_templates),1),1);
        for curr_template_idx = 1:length(good_templates_idx)

            long_isi_total = 0;
            isi_ratios = [];
            for curr_bin = find(use_spiking_stat_bins(curr_template_idx,:))
                curr_spike_times = spike_times_timeline( ...
                    spike_times_timeline > spiking_stat_bins(curr_bin) & ...
                    spike_times_timeline < spiking_stat_bins(curr_bin+1) & ...
                    spike_templates == curr_template_idx);
                curr_isi = diff(curr_spike_times);

                long_isi_total = long_isi_total + sum(curr_isi(curr_isi > 2));

                isi_ratios = [isi_ratios;(2*abs(curr_isi(2:end) - curr_isi(1:end-1)))./ ...
                    (curr_isi(2:end) + curr_isi(1:end-1))];
            end

            prop_long_isi(curr_template_idx) = long_isi_total/ ...
                (sum(use_spiking_stat_bins(curr_template_idx,:))*spiking_stat_window);
            cv2(curr_template_idx) = nanmean(isi_ratios);

        end


        % Cortical classification (like Bartho JNeurophys 2004)
        waveform_duration_cutoff = 400;
        %     narrow = non_str_templates & templateDuration_us <= waveform_duration_cutoff;
        %     wide = non_str_templates & templateDuration_us > waveform_duration_cutoff;

        % Striatum classification
        prop_long_isi_cutoff = 0.35;
        cv2_cutoff = 0.8;

        msn = str_good_templates & ...
            templateDuration_us(good_templates) > waveform_duration_cutoff & ...
            prop_long_isi >= prop_long_isi_cutoff;

        fsi = str_good_templates & ...
            templateDuration_us(good_templates) <= waveform_duration_cutoff & ...
            prop_long_isi < prop_long_isi_cutoff;

        tan = str_good_templates & ...
            templateDuration_us(good_templates) > waveform_duration_cutoff & ...
            prop_long_isi < prop_long_isi_cutoff;

        uin = str_good_templates & ~msn & ~fsi & ~tan;

        disp(['Found ' num2str(sum(msn)) ' MSNs' newline ...
            'Found ' num2str(sum(fsi)) ' FSIs' newline ...
            'Found ' num2str(sum(tan)) ' TANs' newline ...
            'Found ' num2str(sum(uin)) ' UINs'])

        waveform_t = 1e3*((0:size(templates,2)-1)/ephys_sample_rate);

        %     figure; hold on;
        %     p = plot(waveform_t,waveforms(str_templates,:)');
        %     set(p(msn(str_templates)),'color','m')
        %     set(p(fsi(str_templates)),'color','b')
        %     set(p(tan(str_templates)),'color','g')
        %     set(p(uin(str_templates)),'color','c')
        %     xlabel('Time (ms)')
        %     title('Striatum');
        %     legend([p(find(msn(str_templates),1)),p(find(fsi(str_templates),1)), ...
        %         p(find(tan(str_templates),1)),p(find(uin(str_templates),1))],{'MSN','FSI','TAN','UIN'});

        %% save data in big struct
        struct_ephys_data(use_rec-missing_ephys).rec_day = rec_day;

        struct_ephys_data(use_rec-missing_ephys).good_templates = good_templates;
        struct_ephys_data(use_rec-missing_ephys).template_depths = template_depths;
        struct_ephys_data(use_rec-missing_ephys).spike_templates = spike_templates;
        struct_ephys_data(use_rec-missing_ephys).spike_times_timeline = spike_times_timeline;

%         struct_ephys_data(use_rec-missing_ephys).contra_stimOn_times = contra_stimOn_times;
        struct_ephys_data(use_rec-missing_ephys).stimOn_times = stimOn_times;
        struct_ephys_data(use_rec-missing_ephys).contra_good_trials = contra_good_trials;

        %
        %         struct_ephys_data(use_rec-missing_ephys).msn = msn;
        %         struct_ephys_data(use_rec-missing_ephys).fsi = fsi;
        %         struct_ephys_data(use_rec-missing_ephys).tan = tan;
        %         struct_ephys_data(use_rec-missing_ephys).uin = uin;

        struct_ephys_data(use_rec-missing_ephys).bin_edges = bin_edges;
        struct_ephys_data(use_rec-missing_ephys).bin_centres = bin_centres;

        struct_ephys_data(use_rec-missing_ephys).bin_window_for_sharp = bin_window_for_sharp;
        struct_ephys_data(use_rec-missing_ephys).bin_window_for_wide = bin_window_for_wide;

        struct_ephys_data(use_rec-missing_ephys).str_depth = str_depth;
        struct_ephys_data(use_rec-missing_ephys).str_spikes_in_stim_time = str_spikes_in_stim_time;

        struct_ephys_data(use_rec-missing_ephys).contra_good_trials = contra_good_trials;
        struct_ephys_data(use_rec-missing_ephys).contra_all_spikes_in_stim_time = contra_all_spikes_in_stim_time;
        struct_ephys_data(use_rec-missing_ephys).contra_sharp_p_units = contra_sharp_p_units;
        struct_ephys_data(use_rec-missing_ephys).contra_wide_p_units = contra_wide_p_units;

        struct_ephys_data(use_rec-missing_ephys).centre_good_trials = centre_good_trials;
        struct_ephys_data(use_rec-missing_ephys).centre_all_spikes_in_stim_time = centre_all_spikes_in_stim_time;
        struct_ephys_data(use_rec-missing_ephys).centre_sharp_p_units = centre_sharp_p_units;
        struct_ephys_data(use_rec-missing_ephys).centre_wide_p_units = centre_wide_p_units;

    end


    passive_ephys_data = struct2table(struct_ephys_data); %, 'AsArray',true);  %convert structure to table
    % take out empty rows
    passive_ephys_data = passive_ephys_data(~cellfun(@isempty, passive_ephys_data.(1)), :);
    %     ephys_data = rmmissing(ephys_data);

    save_name = [animal '_str_ephys_data'];
    save(save_name, "passive_ephys_data", "-v7.3");

end

clear all;


%% load

load('AM021_swr_bhv_data.mat');

save_fig_path = 'C:\Users\amarica\Documents\Lab stuff\Random figs\Long_str_stuff\Passive';
animals = swr_bhv_data.animals;

% animals = {'AM008', 'AP008', 'AP009', 'AM011', 'AM012', ...
%    'AM014', 'AM015', 'AM016', 'AM017'};

for animal_idx=1:length(animals)
    animal = animals{animal_idx};
    load([animal '_str_ephys_data']);

    % find days with both behaviour and ephys

    bhv_days = swr_bhv_data.bhv_days{animal_idx};
    ephys_days = passive_ephys_data.rec_day';

    % remove empty ones
    ephys_days = ephys_days(~cellfun('isempty',ephys_days))

    %     strcmpi(bhv_days, ephys_days)

    %     bhv_and_ephys = ismember(bhv_days, ephys_days);

    days_from_learning = swr_bhv_data.days_from_learning{animal_idx};
    days_from_learning = days_from_learning(ismember(bhv_days, ephys_days));


    %% exploratory plots
    % includes per day plots + across days num cells and max amplitude

    % all
    ephys.exploratory_plots(animal, days_from_learning, passive_ephys_data, save_fig_path, 'All')
end

% % MSNs
% ephys.exploratory_plots(animal, days_from_learning, passive_ephys_data, save_fig_path, 'MSN')
%
% % FSIs
% ephys.exploratory_plots(animal, days_from_learning, passive_ephys_data, save_fig_path, 'FSI')
%
% % TANs
% ephys.exploratory_plots(animal, days_from_learning, passive_ephys_data, save_fig_path, 'TAN')
%
% % UINs
% ephys.exploratory_plots(animal, days_from_learning, passive_ephys_data, save_fig_path, 'UIN')

%% temp depths plot


contra_stim_fig = figure('Position', get(0, 'Screensize'));
tiledlayout('flow');
sgtitle([animal ' Contra responsive cells per depth'])

for use_rec=1:height(passive_ephys_data)

    % get data from struct
    rec_day = passive_ephys_data.rec_day{use_rec};
    template_depths = passive_ephys_data.template_depths{use_rec};
    spike_templates = passive_ephys_data.spike_templates{use_rec};
    str_depth = passive_ephys_data.str_depth{use_rec};

    good_templates = passive_ephys_data.good_templates{use_rec};
    good_templates_idx = find(good_templates);

    contra_sharp_p_units = passive_ephys_data.contra_sharp_p_units{use_rec};
    contra_wide_p_units = passive_ephys_data.contra_wide_p_units{use_rec};

    % get resp cells in this group
    % contra templates
    contra_sharp_responsive_templates = good_templates_idx(contra_sharp_p_units < 0.05);
    contra_wide_responsive_templates = good_templates_idx(contra_wide_p_units < 0.05);

    % contra units
    contra_sharp_responsive_units = find(contra_sharp_p_units < 0.05);
    contra_wide_responsive_units = find(contra_wide_p_units < 0.05);

    % plot
    unit_axes = nexttile;
    set(unit_axes,'YDir','reverse');
    hold on;

    norm_spike_n = mat2gray(log10(accumarray(findgroups(spike_templates),1)+1));
    unit_dots = scatter3(norm_spike_n,template_depths(unique(spike_templates)), ...
        unique(spike_templates),20,'k','filled');

    sharp_responsive_unit_dots = scatter3(norm_spike_n(contra_sharp_responsive_units),template_depths(contra_sharp_responsive_templates), ...
        contra_sharp_responsive_templates,20,'magenta','filled');

    wide_responsive_unit_dots = scatter3(norm_spike_n(contra_wide_responsive_units),template_depths(contra_wide_responsive_templates), ...
        contra_wide_responsive_templates,20,'blue','filled');

    both_responsive_templates = intersect(contra_wide_responsive_templates, contra_sharp_responsive_templates);
    both_responsive_units = intersect(contra_wide_responsive_units, contra_sharp_responsive_units);

    both_responsive_unit_dots = scatter3(norm_spike_n(both_responsive_units),template_depths(both_responsive_templates), ...
        both_responsive_templates,20,'green','filled');

    yline(str_depth, 'red')
    xlim(unit_axes,[-0.1,1]);
    ylim([-50, max(template_depths)+50]);
    ylabel('Depth (\mum)')
    xlabel('Normalized log rate')
    title(['Day ' num2str(days_from_learning(use_rec))])
end

legend([sharp_responsive_unit_dots wide_responsive_unit_dots both_responsive_unit_dots], {'Sharp resp', 'Wide resp', 'Both resp'})

contra_stim_fig_name = [animal '_resp_cells_per_depth_Contra.tif'];
contra_stim_fig_path = fullfile(save_fig_path, contra_stim_fig_name);
saveas(contra_stim_fig, contra_stim_fig_path);

% centre
centre_stim_fig = figure('Position', get(0, 'Screensize'));
tiledlayout('flow');
sgtitle([animal ' Centre responsive cells per depth'])

for use_rec=1:height(passive_ephys_data)

    % get data from struct
    rec_day = passive_ephys_data.rec_day{use_rec};
    template_depths = passive_ephys_data.template_depths{use_rec};
    spike_templates = passive_ephys_data.spike_templates{use_rec};
    str_depth = passive_ephys_data.str_depth{use_rec};

    good_templates = passive_ephys_data.good_templates{use_rec};
    good_templates_idx = find(good_templates);

    centre_sharp_p_units = passive_ephys_data.centre_sharp_p_units{use_rec};
    centre_wide_p_units = passive_ephys_data.centre_wide_p_units{use_rec};

    % get resp cells in this group
    % centre templates
    centre_sharp_responsive_templates = good_templates_idx(centre_sharp_p_units < 0.05);
    centre_wide_responsive_templates = good_templates_idx(centre_wide_p_units < 0.05);

    % centre units
    centre_sharp_responsive_units = find(centre_sharp_p_units < 0.05);
    centre_wide_responsive_units = find(centre_wide_p_units < 0.05);

    % plot
    unit_axes = nexttile;
    set(unit_axes,'YDir','reverse');
    hold on;

    norm_spike_n = mat2gray(log10(accumarray(findgroups(spike_templates),1)+1));
    unit_dots = scatter3(norm_spike_n,template_depths(unique(spike_templates)), ...
        unique(spike_templates),20,'k','filled');

    sharp_responsive_unit_dots = scatter3(norm_spike_n(centre_sharp_responsive_units),template_depths(centre_sharp_responsive_templates), ...
        centre_sharp_responsive_templates,20,'magenta','filled');

    wide_responsive_unit_dots = scatter3(norm_spike_n(centre_wide_responsive_units),template_depths(centre_wide_responsive_templates), ...
        centre_wide_responsive_templates,20,'blue','filled');

    both_responsive_templates = intersect(centre_wide_responsive_templates, centre_sharp_responsive_templates);
    both_responsive_units = intersect(centre_wide_responsive_units, centre_sharp_responsive_units);

    both_responsive_unit_dots = scatter3(norm_spike_n(both_responsive_units),template_depths(both_responsive_templates), ...
        both_responsive_templates,20,'green','filled');


    yline(str_depth, 'red')
    xlim(unit_axes,[-0.1,1]);
    ylim([-50, max(template_depths)+50]);
    ylabel('Depth (\mum)')
    xlabel('Normalized log rate')
    title(['Day ' num2str(days_from_learning(use_rec))])
end

legend([sharp_responsive_unit_dots wide_responsive_unit_dots both_responsive_unit_dots], {'Sharp resp', 'Wide resp', 'Both resp'})

centre_stim_fig_name = [animal '_resp_cells_per_depth_Centre.tif'];
centre_stim_fig_path = fullfile(save_fig_path, centre_stim_fig_name);
saveas(centre_stim_fig, centre_stim_fig_path);
end





