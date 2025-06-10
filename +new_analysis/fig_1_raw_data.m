%% DOMAINS
data_path = '\\qnap-ap001.dpag.ox.ac.uk\APlab\Lab\Papers\Marica_2025\data';
ctx_str_maps_data_path = fullfile(data_path, 'ctx_str_maps.mat');
bhv_data_path = fullfile(data_path, 'bhv.mat');
load(ctx_str_maps_data_path)
load(bhv_data_path)

[unique_animals, ~, unique_animals_idx] = unique(ctx_str_maps.animal);

%% choose animal and ld
use_animal = 14;
use_ld = 0;
use_rec = find(unique_animals_idx == use_animal & bhv.days_from_learning == use_ld);
cortex_kernel = ctx_str_maps.cortex_striatum_map{use_rec};

animal = unique_animals{use_animal};
rec_day = ctx_str_maps.rec_day{use_rec};

%% plot
figure;
tiledlayout('flow');
for kidx=1:size(cortex_kernel, 3)
    nexttile;
    imagesc(cortex_kernel(:,:,kidx));
    axis image;
    clim(max(abs(clim)).*[-1,1]*0.7);
    clim(clim/8)
    ap.wf_draw('cortex','k');
    axis off;
    colormap(ap.colormap('WK', [], 4));
end
sgtitle(['Animal ' num2str(use_animal)])

%% check map labels for this rec - replace with new kmeans code
save_path = '\\qnap-ap001.dpag.ox.ac.uk\APlab\Users\Andrada-Maria_Marica\long_str_ctx_data';
ctx_str_maps_data_path = fullfile(save_path, 'ctx_maps_to_str.mat');
load(ctx_str_maps_data_path)
master_U_fn = fullfile(save_path,'U_master.mat');
load(master_U_fn, 'U_master');

% kmeans ids
num_recordings = size(all_ctx_maps_to_str, 1);
% get number of things from total MUA - after re-run save script
all_cortex_kernel_px = cat(3, all_ctx_maps_to_str.cortex_kernel_px{:});

% run kmeans
rng('default');
rng(0);
num_clusters = 4;
kmeans_starting = mean(cell2mat(permute(cellfun(@(x) x(:,:,round(linspace(1,size(x,3),4))), ...
    all_ctx_maps_to_str.cortex_kernel_px(~cellfun(@isempty, ...
    all_ctx_maps_to_str.cortex_kernel_px)),'uni',false),[2,3,4,1])),4);
[cluster_ids, centroids, sumd] = kmeans(...
    reshape(all_cortex_kernel_px,prod(size(U_master,[1,2])),[])',num_clusters, ...
    'Distance','Correlation','start',reshape(kmeans_starting,[],num_clusters)');

% check who the masters are
centroid_images = reshape(centroids, [num_clusters, [size(all_cortex_kernel_px, 1) size(all_cortex_kernel_px, 2)]]);
figure;
tiledlayout('flow');
for idx=1:num_clusters
    nexttile;
    imagesc(squeeze(centroid_images(idx,:,:)))
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    title(['Cluster ', num2str(idx)]);
end
n_depths = arrayfun(@(rec_idx) ...
    size(all_ctx_maps_to_str.depth_group_edges{rec_idx}, 2) -1 * ~isempty(all_ctx_maps_to_str.depth_group_edges{rec_idx}), ...
    1:num_recordings);
per_rec_psth_cluster_ids = mat2cell(cluster_ids, n_depths);
this_rec_cluster_ids = per_rec_psth_cluster_ids{use_rec};

%% MUA and ROI
%% load day
workflow = {'stim_wheel_right*'};
recordings = plab.find_recordings(animal, rec_day, workflow);
rec_time = recordings.recording{end};
verbose = true;
load_parts.behavior = true;
load_parts.ephys = true;
load_parts.widefield = true;
ap.load_recording;

%% define stuff
num_clusters = 3;
smooth_window = 1; % seconds

%% Get striatum MUA in bins of set lengths
AP_longstriatum_find_striatum_depth
num_depths = 3;
depth_group_edges_3 = linspace(striatum_depth(1), striatum_depth(end), num_depths+1);

% Discretize spikes by 3 depths
depth_group = discretize(spike_depths,depth_group_edges_3);

% Get time bins corresponding to widefield frame exposures
% (skip the beginning and end of the recording to avoid artifacts)
sample_rate = (1/mean(diff(wf_t)));
skip_seconds = 60;
time_bins = wf_t(find(wf_t > skip_seconds,1)):1/sample_rate:wf_t(find(wf_t-wf_t(end) < -skip_seconds,1,'last'));
time_bin_centers = time_bins(1:end-1) + diff(time_bins)/2;

% Bin spikes in depth and time
binned_spikes_3 = zeros(max(depth_group),length(time_bins)-1);
for curr_depth = 1:max(depth_group)
    curr_spike_times = spike_times_timelite(depth_group == curr_depth);
    binned_spikes_3(curr_depth,:) = histcounts(curr_spike_times,time_bins);
end

% traces per depth
downsample_ephys = 10;
downsampled_time = downsample(timelite.timestamps, downsample_ephys);
end_timebin = downsampled_time(end)+mean(diff(downsampled_time));
neural_downsampled_time = [downsampled_time; end_timebin];
ephys_bin_window = mean(diff(downsampled_time));

str_mua_trace = cell(num_depths,1);
for curr_depth=1:num_depths
    curr_spike_templates = spike_templates(depth_group ==curr_depth);
    unique_curr_spike_templates = unique(curr_spike_templates);
    str_all_trace = cell2mat(arrayfun(@(unit_idx) histcounts(spike_times_timelite(spike_templates == unit_idx), neural_downsampled_time), ...
        unique_curr_spike_templates, 'UniformOutput',false)) / ephys_bin_window;
    str_mua_trace{curr_depth} = mean(str_all_trace, 1)';
end

%% draw ROIs

% ROIs = cell(num_clusters, 1);
% for cluster_idx=1:num_clusters
%     figure; colormap(gray);
%     imagesc(wf_avg);
%     axis image off
%     clim([0, 20000])
%     ap.wf_draw('ccf', 'y')
%     roi_poly = drawpolygon;
%     ROIs{cluster_idx} = createMask(roi_poly);
%     title(['ROI for Cluster ', num2str(cluster_idx)]);
% end

ROI_colours = {[0.7 0 0]; [0 0 0.7]; [0 0.7 0]};
figure; colormap(gray);
imagesc(wf_avg);
axis image off
clim([0, 20000])
ap.wf_draw('ccf', 'y')
hold on;
% Draw each boundary as a polygon
for  cluster_idx=1:num_clusters
    ROI_bw = ROIs{cluster_idx};
    ROI_pol = bwboundaries(ROI_bw);
    boundary = ROI_pol{1};
    patch(boundary(:,2), boundary(:,1), ROI_colours{cluster_idx}, ...
        'EdgeColor', ROI_colours{cluster_idx}, 'FaceColor', ROI_colours{cluster_idx}, 'LineWidth', 1.5);
    hold on;
end

%% MUA unit plot split probe in 3 

contra_stim_fig = figure('Position', [970   300   520   550]);
tiledlayout('flow');

% plot
unit_axes = nexttile;
set(unit_axes,'YDir','reverse');
hold on;

norm_spike_n = mat2gray(log10(accumarray(findgroups(spike_templates),1)+1));
unit_dots = scatter3(norm_spike_n,template_depths(unique(spike_templates)), ...
    unique(spike_templates),20,'k','filled');

depth_colours = {[0.7 0 0]; [0 0 0.7]; [0 0.7 0]};
for curr_depth=1:num_depths
    x_limits = xlim(unit_axes); % get x-axis limits
    fill_x = [x_limits(1), x_limits(2), x_limits(2), x_limits(1)];
    fill_y_all = [depth_group_edges_3(curr_depth), depth_group_edges_3(curr_depth), ...
        depth_group_edges_3(curr_depth+1), depth_group_edges_3(curr_depth+1)];
    fill(fill_x, fill_y_all, depth_colours{curr_depth}, 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Create shaded box
end

% plot
xlim(unit_axes,[-0.1,1]);
ylim([-50, max(template_depths)+50]);
ylabel('Depth (\mum)', 'FontSize', 38)
xlabel('Normalized log rate', 'FontSize', 38)
ax = get(gca);
% Customize tick labels and spacing
ax.XAxis.FontSize = 30;  
ax.YAxis.FontSize = 30; 
xticks([0, 0.5, 1]); 
yticks([0, 1725, 3500]); 


%% plot MUA
depth_colours = {[0.7 0 0]; [0 0 0.7]; [0 0.7 0]};
figure;
tiledlayout(3,1)
for curr_depth=1:num_depths
    %     ('Position', [680 460 860 520]);
    nexttile;
    smooth_str_mua_trace = smoothdata(str_mua_trace{curr_depth}, 'gaussian', ...
        [1/ephys_bin_window * smooth_window, 0]);
    plot(downsampled_time, smooth_str_mua_trace, 'color', depth_colours{curr_depth})
    hold on;
    xline(stimOn_times, 'k', 'LineWidth', 2)
    xlim([1050 1160])
%     ylim([0 15])
    xlabel('Time (s)', 'FontSize', 34)
    ylabel('Spikes', 'FontSize', 34)
    box off
    AP_scalebar(10, 5)
    title(['Depth ' num2str(curr_depth)])
end

%% get traces for ROIs

ROI_colours = {[0.7 0 0]; [0 0 0.7]; [0 0.7 0]};

figure;
tiledlayout(3,1)
for cluster_idx=1:num_clusters

    roi_trace = ap.wf_roi(wf_U,wf_V,wf_avg,[],ROIs{cluster_idx});
    smooth_roi_trace = smoothdata(roi_trace, 'gaussian', [wf_framerate * smooth_window, 0]);
%     figure('Position', [680 460 860 520]);
    nexttile;
    plot(wf_t, smooth_roi_trace, 'color', ROI_colours{cluster_idx});
    title(['ROI ' num2str(cluster_idx)]);
    xline(stimOn_times, 'k', 'LineWidth', 2)
    ylim([-0.01, 0.02])
    xlim([1050 1160])
    xlabel('Time (s)', 'FontSize', 34)
    ylabel('{\Delta}F/F', 'FontSize', 34)
    box off
    AP_scalebar(10, 0.01)
end




