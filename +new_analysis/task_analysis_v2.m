%% load
save_path = '\\qnap-ap001.dpag.ox.ac.uk\APlab\Users\Andrada-Maria_Marica\long_str_ctx_data';

bhv_data_path = fullfile(save_path, "swr_bhv.mat");
task_wf_data_path = fullfile(save_path, "task_ctx_wf.mat");
task_ephys_data_path = fullfile(save_path, "task_ephys.mat");
ctx_str_maps_data_path = fullfile(save_path, 'ctx_maps_to_str.mat');

load(bhv_data_path)
load(task_wf_data_path)
load(task_ephys_data_path)
load(ctx_str_maps_data_path)
%
master_U_fn = fullfile(save_path,'U_master.mat');
load(master_U_fn, 'U_master');

%% kmeans ids

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

% %% check the nan cluster ids
% 
% nan_cluster_ids = find(isnan(cluster_ids));
% figure;
% tiledlayout('flow')
% for i=1:length(nan_cluster_ids)
%     nexttile;
%     imagesc(all_cortex_kernel_px(:,:,nan_cluster_ids(i)))
%     axis image;
%     axis off;
%     clim(max(abs(clim)).*[-1,1]*0.7);
%     ap.wf_draw('ccf','k');
%     colormap(ap.colormap('PWG'));
% end

%% heatmaps

%% - stuff
num_recordings = height(task_ephys);

%% - make big vectors of days from learning and mouse id
n_depths = arrayfun(@(rec_idx) ...
    size(task_ephys.binned_spikes_stim_align{rec_idx}, 3) * ~isempty(task_ephys.binned_spikes_stim_align{rec_idx}), ...
    1:num_recordings);
% all_psth_depth_cell = arrayfun(@(rec_idx) 1:n_depths(rec_idx), 1:num_recordings, 'uni', false);

psth_num_trials = arrayfun(@(rec_idx) ...
    size(task_ephys.binned_spikes_stim_align{rec_idx}, 1) * ~isempty(task_ephys.binned_spikes_stim_align{rec_idx}), ...
    1:num_recordings);

per_rec_psth_cluster_ids = mat2cell(cluster_ids, n_depths);

% OLD? not sure why I did this, the above gives the same answer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % move from depth to cluster
% per_rec_psth_cluster_ids = cell(num_recordings, 1);
% for rec_idx=1:num_recordings
%     cluster = per_rec_cluster_ids{rec_idx};
%     psth_depth = all_psth_depth_cell{rec_idx};
%     this_psth_cluster_ids = zeros(size(psth_depth));
%     this_psth_cluster_ids(~isnan(psth_depth)) = cluster(psth_depth(~isnan(psth_depth)));
%     per_rec_psth_cluster_ids{rec_idx} = this_psth_cluster_ids;
% end
% all_psth_cluster_ids = horzcat(psth_cluster_ids{:});

% test_non_eq_idx = find(cellfun(@(x,y) ~isequalwithequalnans(x,y'), per_rec_cluster_ids, per_rec_psth_cluster_ids));

%%%%%%%%%%%%%%%
%% - move psths from depth to cluster and norm

[grouped_cluster_stim_psths, cluster_groups] = ...
    cellfun(@(spikes,cluster_id) ap.groupfun(@sum,spikes,[],[],cluster_id), ....
    task_ephys.binned_spikes_stim_align,per_rec_cluster_ids,'uni',false);
psth_num_clusters = cellfun(@(x) length(x), cluster_groups); 

psth_stim_time = -0.5:0.001:1;
baseline_idx = psth_stim_time >= -0.2 & psth_stim_time <= 0;
softnorm = 0;
gauss_win = gausswin(51, 3)';
cluster_stim_baseline = cellfun(@(psth) squeeze(mean(mean(psth(:,baseline_idx,:), 2), 1)), ...
    grouped_cluster_stim_psths, 'UniformOutput', false, ...
    'ErrorHandler', @(x, varargin) []);
grouped_cluster_norm_stim_psths = arrayfun(@(rec_idx) ...
    (grouped_cluster_stim_psths{rec_idx} - ...
    repmat(permute(cluster_stim_baseline{rec_idx}, [3, 2, 1]), psth_num_trials(rec_idx), length(psth_stim_time), 1)) ...
    ./ (repmat(permute(cluster_stim_baseline{rec_idx}, [3, 2, 1]), psth_num_trials(rec_idx), length(psth_stim_time), 1) + softnorm), ...
    1:num_recordings, ...
    'UniformOutput', false)';
grouped_cluster_smooth_norm_stim_psths = cellfun(@(psth) ...
    filter(gauss_win, sum(gauss_win), psth, [], 2), ...
    grouped_cluster_norm_stim_psths, ...
    'UniformOutput', false);

%% - get big matrix for norm psths 
permute_grouped_cluster_smooth_norm_stim_psths = cellfun(@(x) ...
    permute(x, [1 3 2]), ...
    grouped_cluster_smooth_norm_stim_psths, 'UniformOutput', false);
flattened_grouped_cluster_smooth_norm_stim_psths = arrayfun(@(rec_idx) ...
    reshape(permute_grouped_cluster_smooth_norm_stim_psths{rec_idx}, ...
    psth_num_clusters(rec_idx)*psth_num_trials(rec_idx), ...
    size(permute_grouped_cluster_smooth_norm_stim_psths{rec_idx}, 3)), ...
    1:num_recordings, 'uni', false);
cat_smooth_norm_stim_psths = cat(1, flattened_grouped_cluster_smooth_norm_stim_psths{:});


%% - get indices
% put cluster ids into num_trials
for_psth_cluster_ids_cell = arrayfun(@(rec_idx) repelem(cluster_groups{rec_idx},psth_num_trials(rec_idx)), ...
    1:num_recordings, 'uni', false,'ErrorHandler', @(x, varargin) []);
for_psth_cluster_ids = vertcat(for_psth_cluster_ids_cell{:});

% get trial ids
for_psth_trial_ids_cell = arrayfun(@(rec_idx) repmat(1:psth_num_trials(rec_idx), 1, psth_num_clusters(rec_idx))', ...
    1:num_recordings, 'uni', false);
for_psth_trial_ids = vertcat(for_psth_trial_ids_cell{:});

% get days from learning and animal id
for_psth_days_from_learning = repelem(bhv.days_from_learning, psth_num_clusters'.*psth_num_trials);
[~, ~, for_psth_animal_ids] = unique(repelem(bhv.animal, psth_num_clusters'.*psth_num_trials));

psth_use_days = ~isnan(for_psth_days_from_learning);
[unique_cluster_ids, ~, ~] = unique(for_psth_cluster_ids(psth_use_days));
[psth_unique_days_from_learning, ~, ~] = unique(for_psth_days_from_learning(psth_use_days));
[unique_animal_ids, ~, ~] = unique(for_psth_animal_ids(psth_use_days));

% Combine cluster_ids and days into a single matrix for indexing
psth_group_indices_unique_clusters = [for_psth_animal_ids, ...
    for_psth_trial_ids, ...
    for_psth_cluster_ids, ...
    for_psth_days_from_learning];

%% - package in cluster and day
norm_smooth_stim_psths = cell(num_clusters, length(psth_unique_days_from_learning));
for cluster_idx=1:num_clusters
    for day_idx=1:length(psth_unique_days_from_learning)
        norm_smooth_stim_psths{cluster_idx, day_idx} = ...
            cat_smooth_norm_stim_psths(...
            psth_group_indices_unique_clusters(:, 3) == cluster_idx & ...
            psth_group_indices_unique_clusters(:, 4) == psth_unique_days_from_learning(day_idx), :);
    end
end

%% make RT sorting
all_RT = vertcat(bhv.stim_to_move);
for_psth_RT_cell = arrayfun(@(rec_idx) repmat(all_RT{rec_idx},psth_num_clusters(rec_idx), 1), ...
    1:num_recordings, 'uni', false);
for_psth_RT = vertcat(for_psth_RT_cell{:});

RT_cluster_day = cell(num_clusters, length(psth_unique_days_from_learning));
RT_sort_idx = cell(num_clusters, length(psth_unique_days_from_learning));
for cluster_idx=1:num_clusters
    for day_idx=1:length(psth_unique_days_from_learning)
        RT_cluster_day{cluster_idx, day_idx} = ...
            for_psth_RT(...
            psth_group_indices_unique_clusters(:, 3) == cluster_idx & ...
            psth_group_indices_unique_clusters(:, 4) == psth_unique_days_from_learning(day_idx), :);
        [~, RT_sort_idx{cluster_idx, day_idx}] = sort(RT_cluster_day{cluster_idx, day_idx});
    end
end

%% heatmap plot by RT

plot_time = [-0.5 1];
plot_time_idx = psth_stim_time > plot_time(1) & psth_stim_time < plot_time(2);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow')
    for day_idx=1:length(psth_unique_days_from_learning)
        nexttile;
        imagesc(psth_stim_time(plot_time_idx), [], ...
            norm_smooth_stim_psths{cluster_idx, day_idx}...
            (RT_sort_idx{cluster_idx, day_idx}, plot_time_idx));
        max_abs = max(abs(norm_smooth_stim_psths{cluster_idx, day_idx}...
            (RT_sort_idx{cluster_idx, day_idx}, plot_time_idx)), [], 'all');
        hold on;
        plot(RT_cluster_day{cluster_idx, day_idx}(RT_sort_idx{cluster_idx, day_idx}),...
            1:size(RT_cluster_day{cluster_idx, day_idx}, 1))
        if isempty(max_abs)
            continue
        end
%         clim([-max_abs max_abs])
                clim([-2,2]);
        colormap(ap.colormap('BWR'))
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
    end
    sgtitle(['NEW Cluster ' num2str(cluster_idx)])
end


%%%%%%%%%%%%%%%%%%%%%% LEFT HERE %%%%%%%%%%%%%%%%%%%%%%

%% - split day into thirds

%% - split trials into 3
num_split_trials = 3;
temp_for_psth_split_trial_ids_cell = arrayfun(@(rec_idx) repelem(1:num_split_trials, ceil(psth_num_trials(rec_idx) / num_split_trials)), ...
    1:num_recordings, 'UniformOutput', false);
temp_for_psth_split_trial_ids_cell_trimmed = arrayfun(@(rec_idx) temp_for_psth_split_trial_ids_cell{rec_idx}(1:psth_num_trials(rec_idx)), ...
    1:num_recordings, 'UniformOutput', false);
for_psth_split_trial_ids_cell = arrayfun(@(rec_idx) repmat(temp_for_psth_split_trial_ids_cell_trimmed{rec_idx}, 1, ...
    psth_num_clusters(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_split_trial_ids = vertcat(for_psth_split_trial_ids_cell{:});

%% - HERE: do nested groupfun 
% get split trial indices
psth_split_trial_indices = [for_psth_animal_ids, for_psth_split_trial_ids, ...
    for_psth_days_from_learning, for_psth_cluster_ids];

[psth_unique_split_trial_indices, ~, psth_split_trial_indices] = unique(psth_split_trial_indices, 'rows');

% do avg instead of sum, use nested groupfun to cut down both operations
% into one line 
psth_grouped_cluster_split_trial = ap.groupfun(@sum, ...
    cat_flattened_stim_binned_spikes(psth_use_days, :), psth_split_trial_indices, []);

% % norm and smooth split trial
% psth_stim_time = -0.5:0.001:1;
% baseline_idx = psth_stim_time >= -0.2 & psth_stim_time <= 0;
% 
% % Function to normalize and smooth
% gauss_win = gausswin(51, 3)';
% normalize_and_smooth = @(psth) ...
%     filter(gauss_win, sum(gauss_win), (psth - mean(psth(baseline_idx), 2)) ./ mean(psth(baseline_idx), 2), [], 2);
% 
% psth_grouped_cluster_split_trial_smooth_norm = cell2mat(arrayfun(@(rep) ...
%     normalize_and_smooth(psth_grouped_cluster_split_trial(rep, :)), ...
%     1:size(psth_grouped_cluster_split_trial, 1), ...
%     'UniformOutput', false)');

% group by ld and split trial
psth_day_trial_indices = [psth_unique_split_trial_indices(:, 2), psth_unique_split_trial_indices(:, 3), ...
    psth_unique_split_trial_indices(:, 4)];
[psth_unique_avg_animal_group_indices, ~, psth_animal_group_clusters_indices] = unique(psth_day_trial_indices, 'rows');

avg_psth_grouped_split_trial_smooth_norm = ap.groupfun(@mean, ...
    psth_grouped_cluster_split_trial_smooth_norm, psth_animal_group_clusters_indices, []);

% count animals
num_animals_stim_psth = accumarray(psth_animal_group_clusters_indices, 1);

%% plot split trial psths
my_colormap = ap.colormap('KR', num_split_trials);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = psth_unique_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_psth = avg_psth_grouped_split_trial_smooth_norm(this_day_idx & this_cluster_idx, :);

        nexttile;
        plot(psth_stim_time, for_plot_psth);
        ylim([-0.5 4]);
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['Trial split Cluster ' num2str(cluster_idx)])
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
