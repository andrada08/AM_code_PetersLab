%% load
save_path = '\\qnap-ap001.dpag.ox.ac.uk\APlab\Users\Andrada-Maria_Marica\long_str_ctx_data';

bhv_data_path = fullfile(save_path, "swr_bhv.mat");
% wf_data_path = fullfile(save_path, "ctx_wf.mat");
task_ephys_data_path = fullfile(save_path, "task_ephys.mat");
ctx_str_maps_data_path = fullfile(save_path, 'ctx_maps_to_str.mat');

load(bhv_data_path)
% load(wf_data_path)
load(task_ephys_data_path)
load(ctx_str_maps_data_path)
% 
% master_U_fn = fullfile(save_path,'U_master.mat');
% load(master_U_fn, 'U_master');

%% kmeans ids

% get number of things from total MUA - after re-run save script
all_cortex_kernel_px = cat(3, all_ctx_maps_to_str.cortex_kernel_px{:});
all_flattened_cortex_kernel_px = reshape(all_cortex_kernel_px, [], ...
    size(all_cortex_kernel_px, 3))';

% run kmeans
rng('default');
rng(0);
num_clusters = 4;
[cluster_ids, centroids, sumd] = kmeans(double(all_flattened_cortex_kernel_px), num_clusters, 'Distance', 'correlation',  'Replicates',5);


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

%% heatmaps

%% - stuff
num_recordings = height(task_ephys);
% unique_stims_nan = unique(vertcat(task_ephys.trial_stim_values{:})); 
% unique_stims = unique_stims_nan(~isnan(unique_stims_nan));

%% - make big vectors of days from learning and mouse id
% maps_n_depths = arrayfun(@(rec_idx) ...
%     size(all_ctx_maps_to_str.cortex_kernel_px{rec_idx}, 3) * ~isempty(all_ctx_maps_to_str.cortex_kernel_px{rec_idx}), ...
%     1:height(all_ctx_maps_to_str));
% 
% find(maps_n_depths - n_depths)

n_depths = arrayfun(@(rec_idx) ...
    size(task_ephys.binned_spikes_stim_align{rec_idx}, 3) * ~isempty(task_ephys.binned_spikes_stim_align{rec_idx}), ...
    1:num_recordings);
all_psth_depth_cell = arrayfun(@(rec_idx) 1:n_depths(rec_idx), 1:num_recordings, 'uni', false);

per_rec_cluster_ids = mat2cell(cluster_ids, n_depths);

% move from depth to cluster
per_rec_psth_cluster_ids = cell(num_recordings, 1);
for rec_idx=1:num_recordings
    cluster = per_rec_cluster_ids{rec_idx};
    psth_depth = all_psth_depth_cell{rec_idx};
    this_psth_cluster_ids = zeros(size(psth_depth));
    this_psth_cluster_ids(~isnan(psth_depth)) = cluster(psth_depth(~isnan(psth_depth)));
    per_rec_psth_cluster_ids{rec_idx} = this_psth_cluster_ids;
end
% all_psth_cluster_ids = horzcat(psth_cluster_ids{:});

% put cluster ids into num_trials
num_trials = arrayfun(@(rec_idx) ...
    size(task_ephys.binned_spikes_stim_align{rec_idx}, 1) * ~isempty(task_ephys.binned_spikes_stim_align{rec_idx}), ...
    1:num_recordings);
for_psth_cluster_ids_cell = arrayfun(@(rec_idx) repelem(per_rec_psth_cluster_ids{rec_idx}, num_trials(rec_idx))', ...
    1:num_recordings, 'uni', false);
for_psth_cluster_ids = vertcat(for_psth_cluster_ids_cell{:});

% get trial ids
for_psth_trial_ids_cell = arrayfun(@(rec_idx) repelem(1:num_trials(rec_idx), n_depths(rec_idx))', ...
    1:num_recordings, 'uni', false);
for_psth_trial_ids = vertcat(for_psth_trial_ids_cell{:});

% disp(length(for_psth_cluster_ids_all))
% disp(sum(n_depths.*num_trials))

% get days from learning and animal id
for_psth_days_from_learning = repelem(bhv.days_from_learning, n_depths.*num_trials);
[~, ~, for_psth_animal_ids] = unique(repelem(bhv.animal, n_depths.*num_trials));

% get flattened psths
% permute first
permute_stim_binned_spikes = arrayfun(@(rec_idx) ...
    permute(task_ephys.binned_spikes_stim_align{rec_idx}, [1 3 2]), ...
    1:num_recordings, 'uni', false);
flattened_stim_binned_spikes = arrayfun(@(rec_idx) ...
    reshape(permute_stim_binned_spikes{rec_idx}, ...
    n_depths(rec_idx)*num_trials(rec_idx), ...
    size(permute_stim_binned_spikes{rec_idx}, 3)), ...
    1:num_recordings, 'uni', false);
cat_flattened_stim_binned_spikes = cat(1, flattened_stim_binned_spikes{:});

% A_back = reshape(flattened_stim_binned_spikes{2}, num_trials(2), 1501, n_depths(2));

% figure;
% imagesc(task_ephys.binned_spikes_stim_align{2}(:,:,1))
% 
% figure;
% imagesc(A_back(:,:,1))


%% - group psths according to animal, cluster and day from learning
% only use nonnan learning days
use_days = ~isnan(for_psth_days_from_learning);

[unique_cluster_ids, ~, ~] = unique(for_psth_cluster_ids(use_days));
[unique_days_from_learning, ~, ~] = unique(for_psth_days_from_learning(use_days));
[unique_animal_ids, ~, ~] = unique(for_psth_animal_ids(use_days));

% Combine cluster_ids and days into a single matrix for indexing
group_indices = [for_psth_trial_ids(use_days), for_psth_animal_ids(use_days), for_psth_cluster_ids(use_days), for_psth_days_from_learning(use_days)];
[group_indices_unique_clusters, ~, group_clusters_indices] = unique(group_indices, 'rows');

grouped_cluster_stim_psths = ap.groupfun(@sum, ...
        cat_flattened_stim_binned_spikes(use_days, :), group_clusters_indices, []);


%% norm and smooth
time_vector = -0.5:0.001:1;
baseline_idx = time_vector >= -0.2 & time_vector <= 0;

% Function to normalize and smooth
gauss_win = gausswin(51, 3)';
normalize_and_smooth = @(psth) ...
    filter(gauss_win, sum(gauss_win), (psth - mean(psth(baseline_idx), 2)) ./ mean(psth(baseline_idx), 2), [], 2);

grouped_cluster_smooth_norm_stim_psths = cell2mat(arrayfun(@(rep) ...
    normalize_and_smooth(grouped_cluster_stim_psths(rep, :)), ...
    1:size(grouped_cluster_stim_psths, 1), ...
    'UniformOutput', false)');

%% package in cluster and day

norm_smooth_stim_psths = cell(num_clusters, length(unique_days_from_learning));
for cluster_id=1:num_clusters
    for day_idx=1:length(unique_days_from_learning)
        norm_smooth_stim_psths{cluster_id, day_idx} = ...
            grouped_cluster_smooth_norm_stim_psths(...
            group_indices_unique_clusters(:, 3) == cluster_id & ...
            group_indices_unique_clusters(:, 4) == unique_days_from_learning(day_idx), :);
    end
end


%% plot sorted by resp
sort_time = time_vector > 0 & time_vector < 0.2;
sorted_norm_smooth_stim_psths = cell(num_clusters, length(unique_days_from_learning));
for cluster_id=1:num_clusters
    for day_idx=1:length(unique_days_from_learning)
        [~, sort_idx] = sort(mean(norm_smooth_stim_psths{cluster_id, day_idx}(:, sort_time), 2), 'descend');
        sorted_norm_smooth_stim_psths{cluster_id, day_idx} = norm_smooth_stim_psths{cluster_id, day_idx}(sort_idx, :);
    end
end

plot_time = [-0.5 1];
plot_time_idx = time_vector > plot_time(1) & time_vector < plot_time(2);
for cluster_id=1:num_clusters
    figure;
    tiledlayout('flow')
    for day_idx=1:length(unique_days_from_learning)
        nexttile;
        imagesc(time_vector(plot_time_idx), [], sorted_norm_smooth_stim_psths{cluster_id, day_idx}(:, plot_time_idx))
        max_abs = max(abs(sorted_norm_smooth_stim_psths{cluster_id, day_idx}(:, plot_time_idx)), [], 'all');
        if isempty(max_abs)
            continue
        end
        clim([-max_abs max_abs])
        colormap(ap.colormap('BWR'))
        title(['Day ' num2str(unique_days_from_learning(day_idx))])
    end
    sgtitle(['Cluster ' num2str(cluster_id)])
end

% test = group_indices_unique_clusters(:,[1,3]);
% size(unique(test, 'rows'))
% 
% sum(cellfun(@(x) ~isempty(x), ephys.trial_stim_values) & ~isnan(bhv.days_from_learning))

%% make RT sorting
all_RT = vertcat(bhv.stim_to_move);

% empty the ones we don't use
all_RT(num_trials == 0) = [];

vertcat(all_RT)


for_psth_trial_ids_cell = arrayfun(@(rec_idx) repelem(1:num_trials(rec_idx), n_depths(rec_idx))', ...
    1:num_recordings, 'uni', false);
for_psth_trial_ids = vertcat(for_psth_trial_ids_cell{:});


