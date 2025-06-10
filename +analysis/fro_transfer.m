
%% Load dataset

animal = 'AM021';
% rec_day = '2024-07-23';
% rec_time = '1340';
workflow = {'lcr_passive'};
recordings = plab.find_recordings(animal, [], workflow);

use_rec = length(recordings)-1;
rec_day = recordings(use_rec).day;
rec_time = recordings(use_rec).recording{end};
verbose = true;
load_parts.behavior = true;
load_parts.ephys = true;
load_parts.widefield = true;
ap.load_recording;


%% Get striatum MUA in bins of set lengths

% Set MUA bin length (microns)
mua_length = 200;

% Get striatum start = lowest unit density, end = end of probe
% (this probably isn't robust/optimal)
unit_density_bins = 0:100:3840;
unit_density = histcounts(template_depths,unit_density_bins);
[~,unit_density_min_bottom_idx] = min(fliplr(unit_density));
unit_density_min_idx = length(unit_density_bins) - unit_density_min_bottom_idx;
template_depths_sorted = sort(template_depths);
str_start =  template_depths_sorted(find(template_depths_sorted >= ...
    unit_density_bins(unit_density_min_idx+1),1));
str_end = max(channel_positions(:,2));

% Discretize spikes by depth
depth_group_edges = str_start:mua_length:str_end;
depth_group = discretize(spike_depths,depth_group_edges);

% Get time bins corresponding to widefield frame exposures
% (skip the beginning and end of the recording to avoid artifacts)
sample_rate = (1/mean(diff(wf_t)));
skip_seconds = 60;
time_bins = wf_t(find(wf_t > skip_seconds,1)):1/sample_rate:wf_t(find(wf_t-wf_t(end) < -skip_seconds,1,'last'));
time_bin_centers = time_bins(1:end-1) + diff(time_bins)/2;

% Bin spikes in depth and time
binned_spikes = zeros(max(depth_group),length(time_bins)-1);
for curr_depth = 1:max(depth_group)
    curr_spike_times = spike_times_timelite(depth_group == curr_depth);
    binned_spikes(curr_depth,:) = histcounts(curr_spike_times,time_bins);
end

%% Regress cortical fluorescence to striatal MUA

% Normalize MUA by standard devation
% (to have kernels in comparable units)
binned_spikes_std = binned_spikes./nanstd(binned_spikes,[],2);

% Set parameters for regression
% (widefield components to use - too many can overload memory or add noise,
% too few gives a less detailed map)
use_svs = 1:500;
% (frame lags to use: for maps rather than prediction, can just do zero-lag
% regression with no temporal component)
kernel_t = [0,0];
kernel_frames = round(kernel_t(1)*sample_rate):round(kernel_t(2)*sample_rate);
% (ridge regression normalizing factor lambda: too low gives just noise,
% higher removes map detail)
lambda = 5;
% (whether to z-score regressors or signals: not here, since everything
% already normalized)
zs = [false,false];
% (cross validation: not needed for maps, only for prediction)
cvfold = 1;
% (whether to use (yes) and return as ouput (no) a constant in regression)
use_constant = true;
return_constant = false;

% Interpolate the widefield to the center of each time bin to match the
% MUA timepoints
fVdf_deconv_resample = interp1(wf_t,wf_V(use_svs,:)',time_bin_centers)';

% Do regression from cortex to spikes
[cortex_kernel,predicted_spikes,explained_var] = ...
    ap.regresskernel(fVdf_deconv_resample, ...
    binned_spikes_std,kernel_frames, ...
    lambda,zs,cvfold,return_constant,use_constant);

% (note: cortex_kernel is components x time x MUA - so in this example is
% 500 used components x 1 time point x 12 MUA signals)

% Convert cortical kernal V into pixels, plot
cortex_kernel_px = squeeze(plab.wf.svd2px(wf_U(:,:,use_svs),cortex_kernel));
ap.imscroll(cortex_kernel_px);
axis image;
clim(max(abs(clim)).*[-1,1]*0.7);
ap.wf_draw('ccf','k');
colormap(ap.colormap('PWG'));

%% FOR TRANSFER
%% - plot example map
curr_depth = 1;
figure;
imagesc(cortex_kernel_px(:,:,curr_depth))
axis image;
axis off;
ap.wf_draw('ccf','k');
colormap(ap.colormap('PWG'));
title('Example map');
clim_val = max(abs(clim)) * 0.7;
clim([-clim_val, clim_val]); % Set the color limits
clim(clim/8)

curr_depth = 5;
figure;
imagesc(cortex_kernel_px(:,:,curr_depth))
axis image;
axis off;
ap.wf_draw('ccf','k');
colormap(ap.colormap('PWG'));
title('Example map');
clim_val = max(abs(clim)) * 0.7;
clim([-clim_val, clim_val]); % Set the color limits
clim(clim/8)

curr_depth = 12;
figure;
imagesc(cortex_kernel_px(:,:,curr_depth))
axis image;
axis off;
ap.wf_draw('ccf','k');
colormap(ap.colormap('PWG'));
title('Example map');
clim_val = max(abs(clim)) * 0.7;
clim([-clim_val, clim_val]); % Set the color limits
clim(clim/8)

%% !colorbar!
figure;
imagesc([-clim_val clim_val]); % Dummy image to create colorbar scale
colormap(ap.colormap('PWG'));
colorbar_handle = colorbar;
colorbar_handle.Ticks = [-clim_val, clim_val]; 
colorbar_handle.TickLabels = {'-max', 'max'};   
colorbar_handle.Label.String = 'Weights';       
colorbar_handle.FontSize = 40;
axis off; % Hide axis since this figure is only for colorbar

%% - plot example mua trace

figure;
subplot(2,1,1)
plot(time_bin_centers, binned_spikes_std(curr_depth,:), 'color', [0.7 0 0])
xlabel('Time (s)', 'FontSize', 34)
ylabel('Spikes', 'FontSize', 34)
xlim([75 100])
ylim([-1 9])
box off

subplot(2,1,2)
plot(time_bin_centers, predicted_spikes(curr_depth,:), 'color', [0 0.8 0.8])
xlabel('Time (s)', 'FontSize', 34)
ylabel('Spikes', 'FontSize', 34)
xlim([75 100])
ylim([-1 9])
box off
AP_scalebar(2, 2)
%% - plot depth units for chosen
curr_depth = 8;

contra_stim_fig = figure('Position', [970   300   520   550]);
tiledlayout('flow');
str_depth = [str_start str_end];

% plot
unit_axes = nexttile;

set(unit_axes,'YDir','reverse');
hold on;

norm_spike_n = mat2gray(log10(accumarray(findgroups(spike_templates),1)+1));
unit_dots = scatter3(norm_spike_n,template_depths(unique(spike_templates)), ...
    unique(spike_templates),20,'k','filled');

% curr_spike_templates = spike_templates(depth_group == curr_depth);
% curr_depth_unit_dots = scatter3(norm_spike_n(curr_spike_templates),template_depths(curr_spike_templates), ...
%     curr_spike_templates,20,'filled', 'MarkerFaceColor', [0.7 0 0]);
% 
% x_limits = xlim(unit_axes); % get x-axis limits
% fill_x = [x_limits(1), x_limits(2), x_limits(2), x_limits(1)];
% fill_y_all = [str_depth(1), str_depth(1), str_depth(2), str_depth(2)];
% fill(fill_x, fill_y_all, [0.5 0.5 0.5], 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Create shaded box
% 
% fill_y_section = [depth_group_edges(curr_depth), depth_group_edges(curr_depth), depth_group_edges(curr_depth+1), depth_group_edges(curr_depth+1)];
% fill(fill_x, fill_y_section, 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Create shaded box

xlim(unit_axes,[-0.1,1]);
ylim([-50, max(template_depths)+50]);

axis off 
ylabel('Depth (\mum)', 'FontSize', 38)
xlabel('Normalized log rate', 'FontSize', 38)

ax = get(gca);
% Customize tick labels and spacing
ax.XAxis.FontSize = 30;  % Set X-axis tick label font size
ax.YAxis.FontSize = 30;  % Set Y-axis tick label font size

% Set specific intervals for ticks to make them more sparse
xticks([0, 0.5, 1]); % 5 ticks on X-axis
yticks([0, 1725, 3500]); % Customize Y-axis tick intervals

%% - plot depth units for all striatum

contra_stim_fig = figure('Position', [970   300   520   550]);
tiledlayout('flow');
str_depth = [str_start str_end];

% plot
unit_axes = nexttile;
set(unit_axes,'YDir','reverse');
hold on;

norm_spike_n = mat2gray(log10(accumarray(findgroups(spike_templates),1)+1));
unit_dots = scatter3(norm_spike_n,template_depths(unique(spike_templates)), ...
    unique(spike_templates),20,'k','filled');

curr_spike_templates = spike_templates(depth_group == curr_depth);
% curr_depth_unit_dots = scatter3(norm_spike_n(curr_spike_templates),template_depths(curr_spike_templates), ...
%     curr_spike_templates,20,'filled', 'MarkerFaceColor', [0.7 0 0]);

x_limits = xlim(unit_axes); % get x-axis limits
fill_x = [x_limits(1), x_limits(2), x_limits(2), x_limits(1)];
fill_y_all = [str_depth(1), str_depth(1), str_depth(2), str_depth(2)];
fill(fill_x, fill_y_all, 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Create shaded box

xlim(unit_axes,[-0.1,1]);
ylim([-50, max(template_depths)+50]);
ylabel('Depth (\mum)', 'FontSize', 38)
xlabel('Normalized log rate', 'FontSize', 38)

ax = get(gca);
% Customize tick labels and spacing
ax.XAxis.FontSize = 30;  % Set X-axis tick label font size
ax.YAxis.FontSize = 30;  % Set Y-axis tick label font size

% Set specific intervals for ticks to make them more sparse
xticks([0, 0.5, 1]); % 5 ticks on X-axis
yticks([0, 1725, 3500]); % Customize Y-axis tick intervals
%% - plot mua avg across trials

contra_stim = 90;
trial_stim_values = vertcat(trial_events.values.TrialStimX);
trial_stim_values = trial_stim_values(1:length(stimOn_times));

% stim aligned wheel move
% create matrix of times for stim onset
timestep = 0.01;
start_time = -0.5;
end_time = 1;  %%%%%%% can change to [0, 0.5]
timevec = start_time:timestep:end_time;
stim_frame = (-start_time)*(1/timestep)+1;
time_stimulus = stimOn_times+timevec;
% stim aligned wheel move
t = timelite.timestamps;
stim_wheel_move = interp1(t,+wheel_move,time_stimulus);
no_move_trials = sum(stim_wheel_move(:,stim_frame:end),2)==0;

contra_good_trials = (trial_stim_values == contra_stim) & no_move_trials;

psth_opts.window = [-0.5,1];
psth_opts.bin_size = 0.001;

binned_spikes_stim_align = ap.ephys_psth(spike_times_timelite(~isnan(depth_group)), ...
    num2cell(stimOn_times),  ...
    depth_group(~isnan(depth_group)));

% around stim time
bin_edges = psth_opts.window(1):psth_opts.bin_size:psth_opts.window(2);
bin_centres = bin_edges(1:end-1) + diff(bin_edges)/2;

% contra stim
contra_no_move_binned_spikes_stim_align = binned_spikes_stim_align(:,:, contra_good_trials);
contra_no_move_psth_stim_align = squeeze(nanmean(contra_no_move_binned_spikes_stim_align, 3));
% - normalize
contra_stim_baseline = mean(contra_no_move_psth_stim_align(:,bin_centres>-0.2 & bin_centres<0), 2);
contra_norm_no_move_psth_stim_align = (contra_no_move_psth_stim_align - contra_stim_baseline) ...
    ./ (contra_stim_baseline + std(contra_stim_baseline));

gauss_win = gausswin(51, 3)';
contra_smooth_norm_no_move_psth_stim_align = filter(gauss_win,sum(gauss_win),contra_norm_no_move_psth_stim_align, [], 2);

all_depths_contra_smooth_norm_stim_align = mean(contra_smooth_norm_no_move_psth_stim_align, 1);

figure('Position', [680 460 860 520]);
plot(bin_edges, all_depths_contra_smooth_norm_stim_align, 'color', [0.7 0 0])
xline(0, 'k', 'LineWidth', 2)
xline(0.5, 'k', 'LineWidth', 2)
xlabel('Time from stim onset (s)', 'FontSize', 34)
ylabel({'Increase in firing rate'; '(spks/s)'}, 'FontSize', 34)
box off
AP_scalebar(0.2, 0.2)
% title('Striatum psth eg day')

%% - plot mua and stim onsets

% curr depth
downsampled_time = downsample(timelite.timestamps, 10);
end_timebin = downsampled_time(end)+mean(diff(downsampled_time));
neural_downsampled_time = [downsampled_time; end_timebin];
bin_window = mean(diff(downsampled_time));

unique_curr_spike_templates = unique(curr_spike_templates);
str_1_all_trace = cell2mat(arrayfun(@(unit_idx) histcounts(spike_times_timelite(spike_templates == unit_idx), neural_downsampled_time), ...
    unique_curr_spike_templates, 'UniformOutput',false)) / bin_window;
str_1_mua_trace = mean(str_1_all_trace, 1)';

gauss_win = gausswin(51, 3)';
smooth_str_1_mua_trace = filter(gauss_win,sum(gauss_win),str_1_mua_trace, [], 1);

% figure; plot(smooth_str_1_mua_trace)
% 
% figure;
% plot(downsampled_time, smooth_str_1_mua_trace)
% hold on;
% xline(stimOn_times)

% all
str_all_trace = histcounts(spike_times_timelite, neural_downsampled_time) / bin_window;
str_mua_trace = mean(str_all_trace, 1)';

smooth_str_mua_trace = filter(gauss_win,sum(gauss_win),str_mua_trace, [], 1);

% figure; plot(smooth_str_mua_trace)

figure('Position', [680 460 860 520]);
plot(downsampled_time, smooth_str_mua_trace, 'color', [0.7 0 0])
hold on;
xline(stimOn_times(contra_good_trials), 'k', 'LineWidth', 2)
xlim([60 120])
xlabel('Time (s)', 'FontSize', 34)
ylabel('Spikes', 'FontSize', 34)
box off
AP_scalebar(10, 500)

%% - wf all trials over vis 

% new ccf and poly drawing
figure('Color', 'white');
ap.wf_draw('ccf', 'k')
set(gca, 'YDir', 'Reverse')
axis off
axis image
box off
roi_poly = drawpolygon;

% with brain
figure; colormap(gray);
imagesc(wf_avg); 
axis image off
clim([0, 20000])
ap.wf_draw('ccf', 'y')
roi_poly = drawpolygon;
vis_roi_mask = createMask(roi_poly);
% figure; imagesc(vis_roi_mask); axis image;
% title('ROI mask');
vis_roi_trace = ap.wf_roi(wf_U,wf_V,wf_avg,[],vis_roi_mask);

figure('Position', [680 460 860 520]);
plot(wf_t, vis_roi_trace, 'color', [0 0.7 0]);
% title('Visual ROI fluorescence');
xline(stimOn_times(contra_good_trials), 'k', 'LineWidth', 2)
ylim([-0.02, 0.02])
xlim([60 120])
xlabel('Time (s)', 'FontSize', 34)
ylabel('{\Delta}F/F', 'FontSize', 34)
box off
AP_scalebar(10, 0.01)

%% - wf avg trials
roi_stim_align = interp1(wf_t,vis_roi_trace,time_stimulus)';
avg_contra_roi_stim_align = mean(roi_stim_align(:, contra_good_trials), 2);
figure('Position', [680 460 860 520]);
plot(timevec, avg_contra_roi_stim_align, 'color', [0 0.7 0])
xline(0, 'k', 'LineWidth', 2)
xline(0.5, 'k', 'LineWidth', 2)
xlabel('Time from stim onset (s)', 'FontSize', 34)
ylabel('{\Delta}F/F', 'FontSize', 34)
% title('Visual ROI fluorescence');
box off
AP_scalebar(0.2, 1*10^-3)


%% TASK
animal = 'AM021';
load([animal '_task_regression_data']);

kernels_across_days_fig = figure('Position', get(0, 'Screensize'));
tiledlayout(4,1)
sgtitle([animal ' Kernels for last day'], 'FontSize', 34, 'FontWeight', 'bold')

kernel_stim = nexttile;
plot(0:49, regression.str_1_stim_move_rew_avg_coeff_stim{end}', ...
    'linewidth', 2, 'Color', [0.7804, 0.0824, 0.5216]);
xlim([-50 100])
ylim([-1 4.5])
% ylabel({'Stim'; 'Spike rate(spks/s)'}, 'FontSize', 34)
xline(0, 'LineWidth', 2)
yline(0, '--', 'LineWidth', 2)
box off

kernel_stim_move = nexttile;
plot(-49:100, ...
    regression.str_1_stim_move_rew_avg_coeff_stim_move{end}', ...
    'linewidth', 2, 'Color', [0.0, 0.4784, 0.7451]);
ylim([-1 4.5])
% ylabel({'Stim move'; 'Spike rate(spks/s)'}, 'FontSize', 34)
xline(0, 'LineWidth', 2)
yline(0, '--', 'LineWidth', 2)
box off

kernel_iti_move = nexttile;
plot(-49:100, ...
    regression.str_1_stim_move_rew_avg_coeff_iti_move{end}', ...
    'linewidth', 2, 'Color', [0.1333, 0.5451, 0.1333]);
ylim([-1 4.5])
% ylabel({'ITI move'; 'Spike rate(spks/s)'}, 'FontSize', 34)
xline(0, 'LineWidth', 2)
yline(0, '--', 'LineWidth', 2)
box off

kernel_reward = nexttile;
plot(0:49, regression.str_1_stim_move_rew_avg_coeff_reward{end}', ...
    'linewidth', 2, 'Color', [0.5, 0.0, 0.5]);
xlim([-50 100])
ylim([-1 4.5])
% ylabel({'Reward'; 'Spike rate(spks/s)'}, 'FontSize', 34)
xline(0, 'LineWidth', 2)
yline(0, '--', 'LineWidth', 2)
xlabel({'Time from'; 'event onset (ms)'}, 'FontSize', 34)
box off
AP_scalebar(20, 1)
% 
% kernels_across_days_fig_name = [animal '_kernels_across_days.tif'];
% kernels_across_days_fig_path = fullfile(save_fig_path, kernels_across_days_fig_name);
% saveas(kernels_across_days_fig, kernels_across_days_fig_path);

%% BEHAVIOUR
%% - wheel trace with stim onset and move onset + heatmaps for RT
animal = 'AM021';

load("new_all_swr_bhv_data.mat")

animal_idx = strcmp(swr_bhv_data.animals, animal);
days_from_learning = swr_bhv_data.days_from_learning{animal_idx};
stimwheel_pval = swr_bhv_data.stimwheel_pval(animal_idx, :);
bhv_days = swr_bhv_data.bhv_days{animal_idx};

workflow = {'stim_wheel_right*'};
recordings = plab.find_recordings(animal, [], workflow);

%% -- day 2
use_rec = 2;
rec_day = recordings(use_rec).day;
rec_time = recordings(use_rec).recording{end};
verbose = true;
load_parts.behavior = true;
ap.load_recording;

figure('Position', [680   720   750   260]);
plot(timelite.timestamps, wheel_velocity, 'k')
hold on
xline(stimOn_times, 'r')
ylabel('Wheel velocity', 'FontSize', 16)
xlabel('Time (s)', 'FontSize', 16)
ylim([-4000 6000])
xlim([150 300]) % day 2
% xlim([350 500]) % day before end
AP_scalebar(20, 2000)

align_times = photodiode_times(1:2:end);
surround_time = [-4,4];
surround_sample_rate = 100;
surround_time_points = surround_time(1):1/surround_sample_rate:surround_time(2);
pull_times = align_times + surround_time_points;

stim_aligned_wheel_vel = interp1(timelite.timestamps, ...
    wheel_velocity,pull_times);

stim_aligned_wheel_move = interp1(timelite.timestamps, ...
    +wheel_move,pull_times);

[sorted_RT, sorted_RT_idx] = sort(stim_to_move);

figure('Position',   [370   180   700  580]);
imagesc(surround_time_points,[],abs(stim_aligned_wheel_vel(sorted_RT_idx, :)))
xline(0,'color','r', 'LineWidth', 2);
clim([0, max(abs(clim))])
colormap(ap.colormap('WK',[] ,0.7))
title('Day 2', 'FontSize', 18)
ylabel('Trial sorted by RT', 'FontSize', 38)
xlabel('Time from stim onset (100ms)', 'FontSize', 38)
colorbar_handle = colorbar;
colorbar_handle.Ticks = clim; 
colorbar_handle.TickLabels = {'0', 'max'};
colorbar_handle.FontSize = 30;
box off;

ax = get(gca);
% Customize tick labels and spacing
ax.XAxis.FontSize = 30;  % Set X-axis tick label font size
ax.YAxis.FontSize = 30;  % Set Y-axis tick label font size

% Set specific intervals for ticks to make them more sparse
xticks([-4, -2, 0, 2, 4]); % 5 ticks on X-axis
yticks([20, 40, 60, 80, 100, 120]); % Customize Y-axis tick intervals

% figure;
% imagesc(surround_time_points,[],stim_aligned_wheel_move)
% xline(0,'color','r');
% clim(max(abs(clim)).*[-1,1])
% colormap('gray');
% title('Day 2')

%% -- day 6
use_rec = 6;
rec_day = recordings(use_rec).day;
rec_time = recordings(use_rec).recording{end};
verbose = true;
load_parts.behavior = true;
ap.load_recording;

figure('Position', [680   720   750   260]);
plot(timelite.timestamps, wheel_velocity, 'k')
hold on
xline(stimOn_times, 'r')
ylabel('Wheel velocity', 'FontSize', 16)
xlabel('Time (s)', 'FontSize', 16)
% xlim([150 300]) % day 2
xlim([350 500]) % day 6
ylim([-4000 6000])
AP_scalebar(20, 2000)

align_times = photodiode_times(1:2:end);
surround_time = [-4,4];
surround_sample_rate = 100;
surround_time_points = surround_time(1):1/surround_sample_rate:surround_time(2);
pull_times = align_times + surround_time_points;

stim_aligned_wheel_vel = interp1(timelite.timestamps, ...
    wheel_velocity,pull_times);

stim_aligned_wheel_move = interp1(timelite.timestamps, ...
    +wheel_move,pull_times);

[sorted_RT, sorted_RT_idx] = sort(stim_to_move);


figure('Position',   [370   180   700  580]);
imagesc(surround_time_points,[],abs(stim_aligned_wheel_vel(sorted_RT_idx, :)))
xline(0,'color','r', 'LineWidth', 2);
clim([0, max(abs(clim))])
colormap(ap.colormap('WK',[] ,0.7))
title('Day 6', 'FontSize', 18)
ylabel('Trial sorted by RT', 'FontSize', 38)
xlabel('Time from stim onset (100ms)', 'FontSize', 38)
colorbar_handle = colorbar;
colorbar_handle.Ticks = clim; 
colorbar_handle.TickLabels = {'0', 'max'};
colorbar_handle.FontSize = 30;
box off;

ax = get(gca);
% Customize tick labels and spacing
ax.XAxis.FontSize = 30;  % Set X-axis tick label font size
ax.YAxis.FontSize = 30;  % Set Y-axis tick label font size

% Set specific intervals for ticks to make them more sparse
xticks([-4, -2, 0, 2, 4]); % 5 ticks on X-axis
yticks([20, 40, 60, 80, 100, 120]); % Customize Y-axis tick intervals

% figure;
% imagesc(surround_time_points,[],stim_aligned_wheel_move)
% xline(0,'color','r');
% clim(max(abs(clim)).*[-1,1])
% colormap('gray');
% title('Day 6')

%% - RT vs null
load("new_all_swr_bhv_data.mat")

animals = swr_bhv_data.animals;

all_days_from_learning = swr_bhv_data.days_from_learning;
all_rxn_null_med = swr_bhv_data.stimwheel_rxn_null_med;
all_rxn_med = swr_bhv_data.stimwheel_rxn_med;

all_days = cell2mat(all_days_from_learning);      
unique_days = unique(all_days);          

% Step 2: Initialize output cell array (11 x num_unique_days)
aligned_all_rxn_null_med = nan(length(animals), numel(unique_days));
aligned_all_rxn_med = nan(length(animals), numel(unique_days));

% Step 3: Fill in output_matrix based on learning days
for animal_id = 1:length(animals)  % Loop through each animal
    current_days = all_days_from_learning{animal_id};            
    
    % Loop through each day for the current animal
    for day_idx = 1:length(current_days)
        day = current_days(day_idx);                 
        new_day_index = find(unique_days == day);   

        % Store the data for this day in the corresponding position
        if ~isempty(all_rxn_null_med{animal_id, day_idx})
            aligned_all_rxn_null_med(animal_id, new_day_index) = all_rxn_null_med{animal_id, day_idx};
            aligned_all_rxn_med(animal_id, new_day_index) = all_rxn_med{animal_id, day_idx};
        end
    end
end

% choose interval
good_days = unique_days<=3 & unique_days>=-3;

% get avg
mean_aligned_all_rxn_med = nanmean(aligned_all_rxn_med(:, good_days), 1);
mean_aligned_all_rxn_null_med = nanmean(aligned_all_rxn_null_med(:, good_days), 1);

% Calculate SEM
sem_aligned_all_rxn_med = nanstd(aligned_all_rxn_med(:, good_days), 0, 1) ./ sqrt(sum(~isnan(aligned_all_rxn_med(:, good_days)), 1));
sem_aligned_all_rxn_null_med = nanstd(aligned_all_rxn_null_med(:, good_days), 0, 1) ./ sqrt(sum(~isnan(aligned_all_rxn_null_med(:, good_days)), 1));

% do index: diff over sum
performance_index = (aligned_all_rxn_null_med(:, good_days) - aligned_all_rxn_med(:, good_days))./ ...
    (aligned_all_rxn_med(:, good_days)+aligned_all_rxn_null_med(:, good_days));
mean_performance_index = nanmean(performance_index, 1);
sem_performance_index = nanstd(performance_index, 0, 1) ./ sqrt(sum(~isnan(performance_index), 1));

% figure for real vs null
figure('Position', [680   400   690   570]);
hold on;
% Plot mean for reaction med with error bars
errorbar(unique_days(good_days), mean_aligned_all_rxn_med, sem_aligned_all_rxn_med, ...
    'DisplayName', 'Mean Reaction Med', 'Color', 'k', 'CapSize', 0);
set(gca, 'YScale', 'log')
% Create shaded area for SEM of null reaction med
fill([unique_days(good_days), fliplr(unique_days(good_days))], ...
     [mean_aligned_all_rxn_null_med + sem_aligned_all_rxn_null_med, ...
     fliplr(mean_aligned_all_rxn_null_med - sem_aligned_all_rxn_null_med)], ...
     [0.5 0.5 0.5], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
set(gca, 'YScale', 'log')
% Plot mean for null reaction med
plot(unique_days(good_days), mean_aligned_all_rxn_null_med, 'DisplayName', 'Mean Null Reaction Med', 'Color', [0.5 0.5 0.5]);
set(gca, 'YScale', 'log')
xline(0, 'LineWidth', 2, 'color', [0.3, 0.0, 0.5])
% Customize the plot
xlabel('Days from association day', 'FontSize', 38);
ylabel('Reaction Time (s)', 'FontSize', 38);
hold off;
xlim([-3 3])
ax = get(gca);
% Customize tick labels and spacing
ax.XAxis.FontSize = 30;  % Set X-axis tick label font size
ax.YAxis.FontSize = 30;  % Set Y-axis tick label font size
% Set specific intervals for ticks to make them more sparse
xticks(-3:1:3); % 5 ticks on X-axis
yticks(1); % Customize Y-axis tick intervals


% Create figure for performance index with error bars
figure('Position', [680   400   690   570]);
hold on;
% Plot mean performance index with error bars (no caps)
errorbar(unique_days(good_days), mean_performance_index, sem_performance_index, 'DisplayName', 'Mean Performance Index', ...
    'Color', 'k', 'MarkerSize', 5, 'LineWidth', 1, 'CapSize', 0);
xline(0, 'LineWidth', 2, 'color', [0.3, 0.0, 0.5])
% Customize the plot
xlabel('Days from association day', 'FontSize', 38);
ylabel('Performance Index', 'FontSize', 38);
hold off;
xlim([-3 3])
ax = get(gca);
% Customize tick labels and spacing
ax.XAxis.FontSize = 30;  % Set X-axis tick label font size
ax.YAxis.FontSize = 30;  % Set Y-axis tick label font size
% Set specific intervals for ticks to make them more sparse
xticks(-3:1:3); % 5 ticks on X-axis
yticks(0:0.05:0.15); % Customize Y-axis tick intervals

% - eg mouse
animal = 'AM021';
animal_idx = strcmp(swr_bhv_data.animals, animal);
% figure for real vs null
figure('Position',   [370   180   700  580]);
hold on;
% Plot mean for reaction med with error bars
plot(unique_days(good_days), aligned_all_rxn_med(animal_idx, good_days), 'Color', 'k');
set(gca, 'YScale', 'log')
% Plot mean for null reaction med
plot(unique_days(good_days), aligned_all_rxn_null_med(animal_idx, good_days), 'Color', [0.5 0.5 0.5]);
set(gca, 'YScale', 'log')
% Customize the plot
xlabel('Days from association day', 'FontSize', 38);
ylabel('log Reaction Time (s)', 'FontSize', 38);
xline(0, 'LineWidth', 2, 'color', [0.3, 0.0, 0.5])
hold off;
xlim([-3 3])
ax = get(gca);
% Customize tick labels and spacing
ax.XAxis.FontSize = 30;  % Set X-axis tick label font size
ax.YAxis.FontSize = 30;  % Set Y-axis tick label font size
% Set specific intervals for ticks to make them more sparse
xticks(-3:1:3); % 5 ticks on X-axis
yticks(1); % Customize Y-axis tick intervals

