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
for_psth_cluster_ids_cell = arrayfun(@(rec_idx) repelem(per_rec_psth_cluster_ids{rec_idx},num_trials(rec_idx))', ...
    1:num_recordings, 'uni', false);
for_psth_cluster_ids = vertcat(for_psth_cluster_ids_cell{:});

% get trial ids
for_psth_trial_ids_cell = arrayfun(@(rec_idx) repmat(1:num_trials(rec_idx), 1, n_depths(rec_idx))', ...
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
psth_group_indices = [for_psth_trial_ids(use_days), for_psth_animal_ids(use_days), for_psth_cluster_ids(use_days), for_psth_days_from_learning(use_days)];
[group_indices_unique_clusters, ~, group_clusters_indices] = unique(psth_group_indices, 'rows');

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
for cluster_idx=1:num_clusters
    for day_idx=1:length(unique_days_from_learning)
        norm_smooth_stim_psths{cluster_idx, day_idx} = ...
            grouped_cluster_smooth_norm_stim_psths(...
            group_indices_unique_clusters(:, 3) == cluster_idx & ...
            group_indices_unique_clusters(:, 4) == unique_days_from_learning(day_idx), :);
    end
end


%% plot sorted by resp
sort_time = time_vector > 0 & time_vector < 0.2;
sorted_norm_smooth_stim_psths = cell(num_clusters, length(unique_days_from_learning));
for cluster_idx=1:num_clusters
    for day_idx=1:length(unique_days_from_learning)
        [~, sort_idx] = sort(mean(norm_smooth_stim_psths{cluster_idx, day_idx}(:, sort_time), 2), 'descend');
        sorted_norm_smooth_stim_psths{cluster_idx, day_idx} = norm_smooth_stim_psths{cluster_idx, day_idx}(sort_idx, :);
    end
end

plot_time = [-0.5 1];
plot_time_idx = time_vector > plot_time(1) & time_vector < plot_time(2);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow')
    for day_idx=1:length(unique_days_from_learning)
        nexttile;
        imagesc(time_vector(plot_time_idx), [], sorted_norm_smooth_stim_psths{cluster_idx, day_idx}(:, plot_time_idx))
        max_abs = max(abs(sorted_norm_smooth_stim_psths{cluster_idx, day_idx}(:, plot_time_idx)), [], 'all');
        if isempty(max_abs)
            continue
        end
        clim([-max_abs max_abs])
        colormap(ap.colormap('BWR'))
        title(['Day ' num2str(unique_days_from_learning(day_idx))])
    end
    sgtitle(['Cluster ' num2str(cluster_idx)])
end

% test = group_indices_unique_clusters(:,[1,3]);
% size(unique(test, 'rows'))
% 
% sum(cellfun(@(x) ~isempty(x), ephys.trial_stim_values) & ~isnan(bhv.days_from_learning))

%% make RT sorting
all_RT = vertcat(bhv.stim_to_move);
temp_RT = arrayfun(@(rec_idx) repmat(all_RT{rec_idx},n_depths(rec_idx), 1), ...
    1:num_recordings, 'uni', false);
for_psth_RT = vertcat(temp_RT{:});

grouped_cluster_RT = ap.groupfun(@mean, ...
        for_psth_RT(use_days), group_clusters_indices, []);

RT_cluster_day = cell(num_clusters, length(unique_days_from_learning));
RT_sort_idx = cell(num_clusters, length(unique_days_from_learning));
for cluster_idx=1:num_clusters
    for day_idx=1:length(unique_days_from_learning)
        RT_cluster_day{cluster_idx, day_idx} = ...
            grouped_cluster_RT(...
            group_indices_unique_clusters(:, 3) == cluster_idx & ...
            group_indices_unique_clusters(:, 4) == unique_days_from_learning(day_idx), :);
        [~, RT_sort_idx{cluster_idx, day_idx}] = sort(RT_cluster_day{cluster_idx, day_idx});
    end
end

% sort_time = time_vector > 0 & time_vector < 0.2;
% sorted_norm_smooth_stim_psths = cell(num_clusters, length(unique_days_from_learning));
% for cluster_id=1:num_clusters
%     for day_idx=1:length(unique_days_from_learning)
%         [~, sort_idx] = sort(mean(norm_smooth_stim_psths{cluster_id, day_idx}(:, sort_time), 2), 'descend');
%         sorted_norm_smooth_stim_psths{cluster_id, day_idx} = norm_smooth_stim_psths{cluster_id, day_idx}(sort_idx, :);
%     end
% end

%% plot by RT

plot_time = [-0.5 1];
plot_time_idx = time_vector > plot_time(1) & time_vector < plot_time(2);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow')
    for day_idx=1:length(unique_days_from_learning)
        nexttile;
        imagesc(time_vector(plot_time_idx), [], ...
            norm_smooth_stim_psths{cluster_idx, day_idx}...
            (RT_sort_idx{cluster_idx, day_idx}, plot_time_idx));
        max_abs = max(abs(norm_smooth_stim_psths{cluster_idx, day_idx}...
            (RT_sort_idx{cluster_idx, day_idx}, plot_time_idx)), [], 'all');
        hold on;
        plot(RT_cluster_day{cluster_idx, day_idx}(RT_sort_idx{cluster_idx, day_idx}),1:size(RT_cluster_day{cluster_idx, day_idx}, 1))
        if isempty(max_abs)
            continue
        end
%         clim([-max_abs max_abs])
        clim([-2,2]);
        colormap(ap.colormap('BWR'))
        title(['Day ' num2str(unique_days_from_learning(day_idx))])
    end
    sgtitle(['Cluster ' num2str(cluster_idx)])
end

%% WF
%% random
num_recordings = height(wf);
wf_stim_time = wf.wf_stim_time{1};

%% make kernels into ROIs
kernel_ROIs = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    kernel_ROIs{cluster_idx} = mat2gray(max(squeeze(centroid_images(cluster_idx,:,:)),0));
end

%% plot kernel ROIs
for cluster_idx=1:num_clusters
    figure;
    imagesc(kernel_ROIs{cluster_idx})
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    title(['ROI for Cluster ', num2str(cluster_idx)]);
end

%% plot kernel ROIs and make new ones
% manual_kernel_ROIs = cell(num_clusters, 1);
% for cluster_idx=1:num_clusters
%     figure;
%     imagesc(kernel_ROIs{cluster_idx})
%     axis image;
%     axis off;
%     clim(max(abs(clim)).*[-1,1]*0.7);
%     ap.wf_draw('ccf','k');
%     colormap(ap.colormap('PWG'));
%     roi_poly = drawpolygon;
%     manual_kernel_ROIs{cluster_idx} = createMask(roi_poly);
%     title(['ROI for Cluster ', num2str(cluster_idx)]);
% end

%% plot manual ones
for cluster_idx=1:num_clusters
    figure;
    imagesc(manual_kernel_ROIs{cluster_idx})
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    title(['ROI for Cluster ', num2str(cluster_idx)]);
end

%% Get avg Vs and baseline subtract
all_V_stim_align = vertcat(wf.V_stim_align{:});

% get baseline
baseline_idx = wf_stim_time > -0.2 & wf_stim_time < 0;
for_V_stim_baseline = mean(all_V_stim_align(:, baseline_idx, :), 2);

all_norm_V_stim_align = all_V_stim_align - for_V_stim_baseline;

%% group Vs
wf_num_trials = arrayfun(@(rec_idx) ...
    size(wf.V_stim_align{rec_idx}, 1) * ~isempty(wf.V_stim_align{rec_idx}), ...
    1:num_recordings);

% get days from learning and animal id
for_wf_days_from_learning = repelem(bhv.days_from_learning, wf_num_trials);
[~, ~, for_wf_animal_ids] = unique(repelem(bhv.animal, wf_num_trials));

% get trial ids per day
for_wf_trial_ids_cell = arrayfun(@(rec_idx) 1:wf_num_trials(rec_idx), ...
    1:num_recordings, 'uni', false);
for_wf_trial_ids = horzcat(for_wf_trial_ids_cell{:})';

% use days
wf_use_days = ~isnan(for_wf_days_from_learning);
wf_unique_days_from_learning = unique(for_wf_days_from_learning(wf_use_days));

% wf_group_indices = [for_wf_days_from_learning(wf_use_days), for_wf_animal_ids(wf_use_days), for_wf_trial_ids(wf_use_days)];
% [wf_unique_day_group_indices, ~, wf_day_group_clusters_indices] = unique(wf_group_indices, 'rows');

% Vs
good_all_norm_V_stim_align = all_norm_V_stim_align(wf_use_days, :, :);
% temp_all_norm_V_stim_align = good_all_norm_V_stim_align(wf_day_group_clusters_indices, :, :);
reshape_temp_all_norm_V_stim_align = permute(good_all_norm_V_stim_align, [3 2 1]);

%% get manual kernel ROI
all_norm_stim_manual_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    this_ROI = manual_kernel_ROIs{cluster_idx};
    all_norm_stim_manual_kernel_roi{cluster_idx} = squeeze(ap.wf_roi(U_master,reshape_temp_all_norm_V_stim_align, [], [], this_ROI));
end

%% package in day 
grouped_stim_manual_kernel_roi = cell(length(wf_unique_days_from_learning), num_clusters);
for day_idx=1:length(wf_unique_days_from_learning)
    this_day = wf_unique_days_from_learning(day_idx);
    this_day_idx = for_wf_days_from_learning == this_day;
    for cluster_idx=1:num_clusters
        grouped_stim_manual_kernel_roi{day_idx, cluster_idx} = all_norm_stim_manual_kernel_roi{cluster_idx}(:, this_day_idx);
    end
end

%% RT sort
cat_all_RT = vertcat(bhv.stim_to_move{:});
for_wf_RT = cat_all_RT(wf_use_days);
% for_wf_RT = temp_wf_RT(wf_day_group_clusters_indices);

sorted_for_wf_RT_cluster_day = cell(length(wf_unique_days_from_learning), 1);
for_wf_RT_sort_idx = cell(length(wf_unique_days_from_learning), 1);
for day_idx=1:length(wf_unique_days_from_learning)
    this_day = wf_unique_days_from_learning(day_idx);
    for_wf_RT_cluster_day = for_wf_RT(for_wf_days_from_learning == this_day);
    [sorted_for_wf_RT_cluster_day{day_idx}, for_wf_RT_sort_idx{day_idx}] = sort(for_wf_RT_cluster_day);
end

% day_idx = find(wf_unique_days_from_learning == 1);
% figure;
% plot(sorted_for_wf_RT_cluster_day{day_idx}, 'o')
% ylim([-0.5 1])
% title('Processed')
% 
% % temp
% temp_RT_test = bhv.stim_to_move(bhv.days_from_learning == 1);
% figure;
% plot(sort(vertcat(temp_RT_test{:})), 'o')
% ylim([-0.5 1])
% title('Original')

%% plot
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(wf_unique_days_from_learning)
        for_plot_manual_kernel_roi = grouped_stim_manual_kernel_roi{day_idx, cluster_idx}(:, for_wf_RT_sort_idx{day_idx});
        nexttile;
        imagesc(wf_stim_time,[], for_plot_manual_kernel_roi');
        clim_val = max(clim);
        clim([-clim_val, clim_val]);
        colormap(ap.colormap('PWG'))
        hold on;
        plot(sorted_for_wf_RT_cluster_day{day_idx}, 1:size(for_plot_manual_kernel_roi, 2))
        title(['Day ' num2str(wf_unique_days_from_learning(day_idx))])
    end
    sgtitle(['Cluster ' num2str(cluster_idx)])
end


%% - split into 3 

%% - split trials into 3

num_split_trials = 3; 

for_wf_split_trial_ids_cell = arrayfun(@(rec_idx) repelem(1:num_split_trials, ceil(wf_num_trials(rec_idx) / num_split_trials)), ...
    1:num_recordings, 'UniformOutput', false);
for_wf_split_trial_ids_cell_trimmed = arrayfun(@(rec_idx) for_wf_split_trial_ids_cell{rec_idx}(1:wf_num_trials(rec_idx)), ...
    1:num_recordings, 'UniformOutput', false);
for_wf_split_trial_ids = horzcat(for_wf_split_trial_ids_cell_trimmed{:})'; 

%% group Vs by split trial and mouse and day
wf_animal_day_trial_indices = [for_wf_animal_ids(wf_use_days), for_wf_days_from_learning(wf_use_days), for_wf_split_trial_ids(wf_use_days)];
[wf_unique_all_group_indices, ~, wf_all_group_indices] = unique(wf_animal_day_trial_indices, 'rows');

grouped_norm_stim_Vs = ap.groupfun(@mean, ...
    reshape_temp_all_norm_V_stim_align, [], [], wf_all_group_indices);

%% group by ld and split trial
wf_day_trial_indices = [wf_unique_all_group_indices(:, 2), wf_unique_all_group_indices(:, 3)];
[wf_unique_avg_animal_group_indices, ~, wf_animal_group_clusters_indices] = unique(wf_day_trial_indices, 'rows');

avg_grouped_norm_stim_Vs = ap.groupfun(@mean, ...
    grouped_norm_stim_Vs, [], [], wf_animal_group_clusters_indices);

%% count animals
num_animals_stim_wf = accumarray(wf_animal_group_clusters_indices, 1);

%% get psths
all_avg_manual_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    this_ROI = manual_kernel_ROIs{cluster_idx};
    all_avg_manual_kernel_roi{cluster_idx} = squeeze(ap.wf_roi(U_master,avg_grouped_norm_stim_Vs, [], [], this_ROI));
end


%% plot ROIs
my_colormap = ap.colormap('KG', num_split_trials);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(wf_unique_days_from_learning)
        this_day = wf_unique_days_from_learning(day_idx);
        this_day_idx = wf_unique_avg_animal_group_indices(:,1) == this_day;
        for_plot_manual_kernel_roi = all_avg_manual_kernel_roi{cluster_idx}(:, this_day_idx);

        nexttile;
        plot(wf_stim_time, for_plot_manual_kernel_roi);
        title(['Day ' num2str(wf_unique_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['Cluster ' num2str(cluster_idx)])
end


%% get max ampl
all_grouped_manual_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    this_ROI = manual_kernel_ROIs{cluster_idx};
    all_grouped_manual_kernel_roi{cluster_idx} = squeeze(ap.wf_roi(U_master,grouped_norm_stim_Vs, [], [], this_ROI));
end

wf_window_for_max = wf_stim_time>0 & wf_stim_time<0.3;
wf_all_grouped_max_ampl = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    wf_all_grouped_max_ampl{cluster_idx} = max(all_grouped_manual_kernel_roi{cluster_idx}(wf_window_for_max, :), [], 1);
end

% group by ld and split trial
wf_mean_max_ampl = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    wf_mean_max_ampl{cluster_idx} = ap.groupfun(@mean, ...
        wf_all_grouped_max_ampl{cluster_idx}, [], wf_animal_group_clusters_indices);
end

% do sem for errorbar
wf_std_max_ampl = cell(num_clusters, 1);
wf_sem_max_ampl = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    wf_std_max_ampl{cluster_idx} = ap.groupfun(@nanstd, ...
        wf_all_grouped_max_ampl{cluster_idx}, [], wf_animal_group_clusters_indices);
    wf_sem_max_ampl{cluster_idx} = wf_std_max_ampl{cluster_idx}' ./ sqrt(num_animals_stim_wf);
end

%% plot max ampl

curr_color = 'k';

plot_day_values = wf_unique_avg_animal_group_indices(:,1); % Extract x values
plot_trial_values = wf_unique_avg_animal_group_indices(:,2); % Extract y values

spacing = 2; % Controls gap size between groups

x_positions = cell(num_clusters, 1); 
y_positions= cell(num_clusters, 1); 
x_labels_wf_max = cell(num_clusters, 1); 
error_vals = cell(num_clusters, 1); 
valid_xticks = cell(num_clusters, 1); 
valid_xticklabels = cell(num_clusters, 1); 
for cluster_idx = 1:num_clusters
    % Initialize variables for each cluster
    x_positions{cluster_idx} = []; % Reset for each cluster
    y_positions{cluster_idx} = [];
    x_labels_wf_max{cluster_idx} = {};
    error_vals{cluster_idx} = [];

    pos_counter = 1; % Start position counter

    for i = 1:length(unique_plot_day_values)
        indices = find(plot_day_values == unique_plot_day_values(i)); 
        num_points = length(indices);

        % Append the current group of x_positions
        x_positions{cluster_idx} = [x_positions{cluster_idx}, pos_counter:pos_counter+num_points-1];
        y_positions{cluster_idx} = [y_positions{cluster_idx}, wf_mean_max_ampl{cluster_idx}(indices)]; 
        x_labels_wf_max{cluster_idx} = [x_labels_wf_max{cluster_idx}; arrayfun(@(a, b) sprintf('(%d,%d)', a, b), ...
            plot_day_values(indices), plot_trial_values(indices), 'UniformOutput', false)];
        error_vals{cluster_idx} = [error_vals{cluster_idx}, wf_sem_max_ampl{cluster_idx}(indices)'];

        % Insert NaN to break the line before the next group
        if i < length(unique_plot_day_values)
            x_positions{cluster_idx} = [x_positions{cluster_idx}, NaN];
            y_positions{cluster_idx} = [y_positions{cluster_idx}, NaN];
            error_vals{cluster_idx} = [error_vals{cluster_idx}, NaN];
            x_labels_wf_max{cluster_idx} = [x_labels_wf_max{cluster_idx}; {''}]; % Add empty label for NaN
        end

        pos_counter = pos_counter + num_points + spacing; % Add spacing

        valid_xticks{cluster_idx} = x_positions{cluster_idx}(~isnan(x_positions{cluster_idx}));
        valid_xticklabels{cluster_idx} = x_labels_wf_max{cluster_idx}(~isnan(x_positions{cluster_idx}));
    end
end

for cluster_idx = 1:num_clusters
    % Plot for this cluster
    figure;
    tiledlayout('flow');

    nexttile;
    imagesc(manual_kernel_ROIs{cluster_idx})
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    title(['ROI for Cluster ', num2str(cluster_idx)]);

    nexttile;
    errorbar(x_positions{cluster_idx}, y_positions{cluster_idx}, error_vals{cluster_idx}, '-o', 'CapSize', 0, ...
            'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
    xticks(valid_xticks{cluster_idx}); % Only set valid x-ticks
    xticklabels(valid_xticklabels{cluster_idx}); % Remove NaN labels
    xtickangle(45); % Rotate for better visibility
    sgtitle(['Cluster ' num2str(cluster_idx)]);
end



