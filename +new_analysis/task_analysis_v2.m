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
baseline_idx = psth_stim_time >= -0.3 & psth_stim_time <= 0;
softnorm = 10;
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

% get combined days from learning idx
for_psth_combined_days_from_learning = for_psth_days_from_learning;
for_psth_combined_days_from_learning(for_psth_days_from_learning<-3) = -3;
for_psth_combined_days_from_learning(for_psth_days_from_learning>2) = 2;

psth_use_days = ~isnan(for_psth_combined_days_from_learning);
[unique_cluster_ids, ~, ~] = unique(for_psth_cluster_ids(psth_use_days));
[psth_unique_combined_days_from_learning, ~, ~] = unique(for_psth_combined_days_from_learning(psth_use_days));
[unique_animal_ids, ~, ~] = unique(for_psth_animal_ids(psth_use_days));

% Combine cluster_ids and days into a single matrix for indexing
psth_group_indices_unique_clusters = [for_psth_animal_ids, ...
    for_psth_trial_ids, ...
    for_psth_cluster_ids, ...
    for_psth_combined_days_from_learning];

%% - package in cluster and day
pakaged_norm_smooth_stim_psths = cell(num_clusters, length(psth_unique_combined_days_from_learning));
for cluster_idx=1:num_clusters
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        pakaged_norm_smooth_stim_psths{cluster_idx, day_idx} = ...
            cat_smooth_norm_stim_psths(...
            psth_group_indices_unique_clusters(:, 3) == cluster_idx & ...
            psth_group_indices_unique_clusters(:, 4) == psth_unique_combined_days_from_learning(day_idx), :);
    end
end

%% make RT sorting
all_RT = vertcat(bhv.stim_to_move);
for_psth_RT_cell = arrayfun(@(rec_idx) repmat(all_RT{rec_idx},psth_num_clusters(rec_idx), 1), ...
    1:num_recordings, 'uni', false);
for_psth_RT = vertcat(for_psth_RT_cell{:});

RT_cluster_day = cell(num_clusters, length(psth_unique_combined_days_from_learning));
RT_sort_idx = cell(num_clusters, length(psth_unique_combined_days_from_learning));
for cluster_idx=1:num_clusters
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        RT_cluster_day{cluster_idx, day_idx} = ...
            for_psth_RT(...
            psth_group_indices_unique_clusters(:, 3) == cluster_idx & ...
            psth_group_indices_unique_clusters(:, 4) == psth_unique_combined_days_from_learning(day_idx), :);
        [~, RT_sort_idx{cluster_idx, day_idx}] = sort(RT_cluster_day{cluster_idx, day_idx});
    end
end

%% heatmap plot by RT

plot_time = [-0.5 1];
plot_time_idx = psth_stim_time > plot_time(1) & psth_stim_time < plot_time(2);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow')
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        nexttile;
        imagesc(psth_stim_time(plot_time_idx), [], ...
            pakaged_norm_smooth_stim_psths{cluster_idx, day_idx}...
            (RT_sort_idx{cluster_idx, day_idx}, plot_time_idx));
        max_abs = max(abs(pakaged_norm_smooth_stim_psths{cluster_idx, day_idx}...
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
        title(['Day ' num2str(psth_unique_combined_days_from_learning(day_idx))])
    end
    sgtitle(['NEW Cluster ' num2str(cluster_idx)])
end

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

%% - avg across animals in trial groups
[avg_psth_grouped_split_trial_smooth_norm,psth_unique_avg_animal_group_indices] = ...
    ap.nestgroupfun({@mean, @mean},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
num_animals_stim_psth = ...
    mean(ap.nestgroupfun({@mean, @length},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]), ...
    2);

%% plot split trial psths
my_colormap = ap.colormap('KR', num_split_trials);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        this_day = psth_unique_combined_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_psth = avg_psth_grouped_split_trial_smooth_norm(this_day_idx & this_cluster_idx, :);

        nexttile;
        plot(psth_stim_time, for_plot_psth);
        ylim([-0.5 4]);
        title(['Day ' num2str(psth_unique_combined_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['Trial split Cluster ' num2str(cluster_idx)])
end

%% get max ampl
psth_window_for_max = psth_stim_time>0 & psth_stim_time<0.3;
psth_all_trial_split_max_ampl = max(cat_smooth_norm_stim_psths(:, psth_window_for_max), [], 2);

% group by ld and split trial
psth_trial_split_mean_max_ampl = ap.nestgroupfun({@mean, @mean},psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]);

% do sem for errorbar
psth_trial_split_std_max_ampl = ap.nestgroupfun({@mean, @nanstd},psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
psth_trial_split_sem_max_ampl = psth_trial_split_std_max_ampl ./ sqrt(num_animals_stim_psth);

%% plot max ampl

curr_color = 'k';

plot_day_values = psth_unique_avg_animal_group_indices(:,2); % Extract x values
unique_plot_day_values = unique(plot_day_values);
plot_trial_values = psth_unique_avg_animal_group_indices(:,1); % Extract y values
plot_cluster_values = psth_unique_avg_animal_group_indices(:,3);

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
        indices = find(plot_day_values == unique_plot_day_values(i) & plot_cluster_values == cluster_idx);
        num_points = length(indices);

        % Append the current group of x_positions
        x_positions{cluster_idx} = [x_positions{cluster_idx}, pos_counter:pos_counter+num_points-1];
        y_positions{cluster_idx} = [y_positions{cluster_idx}, psth_trial_split_mean_max_ampl(indices)'];
        x_labels_wf_max{cluster_idx} = [x_labels_wf_max{cluster_idx}; arrayfun(@(a, b) sprintf('(%d,%d)', a, b), ...
            plot_day_values(indices), plot_trial_values(indices), 'UniformOutput', false)];
        error_vals{cluster_idx} = [error_vals{cluster_idx}, psth_trial_split_sem_max_ampl(indices)'];

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
    imagesc(squeeze(centroid_images(cluster_idx,:,:)))
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    title(['Cluster ', num2str(cluster_idx)]);

    nexttile;
    errorbar(x_positions{cluster_idx}, y_positions{cluster_idx}, error_vals{cluster_idx}, '-o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
    xticks(valid_xticks{cluster_idx}); % Only set valid x-ticks
    xticklabels(valid_xticklabels{cluster_idx}); % Remove NaN labels
    xtickangle(45); % Rotate for better visibility
    title('PSTH max amplitude mean')
    sgtitle(['Cluster ' num2str(cluster_idx)]);
end


%% RT SORT 
num_RT_split_trials = 5;
temp_for_psth_sort_RT_split_trial_ids_cell = cell(1, num_recordings);
for rec_idx=1:num_recordings
    % (to discretize by RT rank)
    rank_RT_per_rec = tiedrank(all_RT{rec_idx});
    temp_for_psth_sort_RT_split_trial_ids_cell{rec_idx} = discretize(rank_RT_per_rec,...
        round(linspace(1,max(rank_RT_per_rec),num_RT_split_trials+1)))';
end
for_psth_sort_RT_split_trial_ids_cell = arrayfun(@(rec_idx) ...
    repmat(temp_for_psth_sort_RT_split_trial_ids_cell{rec_idx}, 1, psth_num_clusters(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_sort_RT_split_trial_ids = vertcat(for_psth_sort_RT_split_trial_ids_cell{:});

% get sorted RT
psth_per_rec_RT_cell = arrayfun(@(rec_idx) repmat(all_RT{rec_idx}', 1, psth_num_clusters(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
psth_per_rec_RT = vertcat(psth_per_rec_RT_cell{:});

%% - avg across animals in sort split trial groups
[avg_psth_grouped_sort_RT_split_trial_smooth_norm,psth_unique_avg_animal_group_indices] = ...
    ap.nestgroupfun({@mean, @mean},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_sort_RT_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
num_animals_stim_psth_sort_RT_split = ...
    mean(ap.nestgroupfun({@mean, @length},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_sort_RT_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]), ...
    2);

%% plot sort RT split psths
my_colormap = ap.colormap('KR', num_RT_split_trials);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        this_day = psth_unique_combined_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_psth = avg_psth_grouped_sort_RT_split_trial_smooth_norm(this_day_idx & this_cluster_idx, :);

        nexttile;
        plot(psth_stim_time, for_plot_psth);
        ylim([-0.5 4]);
        title(['Day ' num2str(psth_unique_combined_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['RT sort split Cluster ' num2str(cluster_idx)])
end

%% get max ampl for RT sort split
psth_window_for_max = psth_stim_time>0 & psth_stim_time<0.3;
psth_all_trial_split_max_ampl = max(cat_smooth_norm_stim_psths(:, psth_window_for_max), [], 2);

% group by ld and split trial
psth_sort_RT_trial_split_mean_max_ampl = ap.nestgroupfun({@mean, @mean},psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_sort_RT_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]);

% do sem for errorbar
psth_sort_RT_trial_split_std_max_ampl = ap.nestgroupfun({@mean, @nanstd},psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_sort_RT_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
psth_sort_RT_trial_split_sem_max_ampl = psth_sort_RT_trial_split_std_max_ampl ./ sqrt(num_animals_stim_psth_sort_RT_split);

%% plot max ampl

curr_color = 'k';

plot_day_values = psth_unique_avg_animal_group_indices(:,2); % Extract x values
unique_plot_day_values = unique(plot_day_values);
plot_trial_values = psth_unique_avg_animal_group_indices(:,1); % Extract y values
plot_cluster_values = psth_unique_avg_animal_group_indices(:,3);

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
        indices = find(plot_day_values == unique_plot_day_values(i) & plot_cluster_values == cluster_idx);
        num_points = length(indices);

        % Append the current group of x_positions
        x_positions{cluster_idx} = [x_positions{cluster_idx}, pos_counter:pos_counter+num_points-1];
        y_positions{cluster_idx} = [y_positions{cluster_idx}, psth_sort_RT_trial_split_mean_max_ampl(indices)'];
        x_labels_wf_max{cluster_idx} = [x_labels_wf_max{cluster_idx}; arrayfun(@(a, b) sprintf('(%d,%d)', a, b), ...
            plot_day_values(indices), plot_trial_values(indices), 'UniformOutput', false)];
        error_vals{cluster_idx} = [error_vals{cluster_idx}, psth_sort_RT_trial_split_sem_max_ampl(indices)'];

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
    imagesc(squeeze(centroid_images(cluster_idx,:,:)))
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    title(['Cluster ', num2str(cluster_idx)]);

    nexttile;
    errorbar(x_positions{cluster_idx}, y_positions{cluster_idx}, error_vals{cluster_idx}, '-o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
    xticks(valid_xticks{cluster_idx}); % Only set valid x-ticks
    xticklabels(valid_xticklabels{cluster_idx}); % Remove NaN labels
    xtickangle(45); % Rotate for better visibility
    ylim([0 8])
    title('PSTH max amplitude mean')
    sgtitle(['SORT split Cluster ' num2str(cluster_idx)]);
end

%% ADD HERE rest of sort
% add sort split mean RT stuff
% redo those plots

%% RT bin split - regular discretize
discretize_all_RT_bin = [-Inf,0:0.2:0.6,Inf];
num_bin_split_trials = length(discretize_all_RT_bin)-1;
log_discretize_all_RT_bin = [-Inf logspace(-1,1,5) Inf];
temp_for_psth_bin_RT_split_trial_ids_cell = cell(1, num_recordings);
for rec_idx=1:num_recordings
    % (to discretize by RT bin)
    temp_for_psth_bin_RT_split_trial_ids_cell{rec_idx} = discretize(all_RT{rec_idx},discretize_all_RT_bin)';
end
for_psth_bin_RT_split_trial_ids_cell = arrayfun(@(rec_idx) ...
    repmat(temp_for_psth_bin_RT_split_trial_ids_cell{rec_idx}, 1, psth_num_clusters(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_bin_RT_split_trial_ids = vertcat(for_psth_bin_RT_split_trial_ids_cell{:});

%% - avg across animals in bin split trial groups
[avg_psth_grouped_bin_RT_split_trial_smooth_norm,psth_unique_avg_animal_group_indices] = ...
    ap.nestgroupfun({@mean, @mean},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_bin_RT_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
num_animals_stim_psth_bin_RT_split = ...
    mean(ap.nestgroupfun({@mean, @length},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_bin_RT_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]), ...
    2);

%% plot bin RT split psths
my_colormap = ap.colormap('KR', num_bin_split_trials);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        this_day = psth_unique_combined_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_psth = avg_psth_grouped_bin_RT_split_trial_smooth_norm(this_day_idx & this_cluster_idx, :);

        nexttile;
        plot(psth_stim_time, for_plot_psth);
        ylim([-0.5 4]);
        title(['Day ' num2str(psth_unique_combined_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['RT bin split Cluster ' num2str(cluster_idx)])
end

%% get max ampl for RT bin split
psth_window_for_max = psth_stim_time>0 & psth_stim_time<0.3;
psth_all_trial_split_max_ampl = max(cat_smooth_norm_stim_psths(:, psth_window_for_max), [], 2);

% group by ld and split trial
psth_bin_RT_trial_split_mean_max_ampl = ap.nestgroupfun({@mean, @mean},psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_bin_RT_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]);

% do sem for errorbar
psth_bin_RT_trial_split_std_max_ampl = ap.nestgroupfun({@mean, @nanstd},psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_bin_RT_split_trial_ids, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
psth_bin_RT_trial_split_sem_max_ampl = psth_bin_RT_trial_split_std_max_ampl ./ sqrt(num_animals_stim_psth_bin_RT_split);

%% plot max ampl

curr_color = 'k';

plot_day_values = psth_unique_avg_animal_group_indices(:,2); % Extract x values
unique_plot_day_values = unique(plot_day_values);
plot_trial_values = psth_unique_avg_animal_group_indices(:,1); % Extract y values
plot_cluster_values = psth_unique_avg_animal_group_indices(:,3);

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
        indices = find(plot_day_values == unique_plot_day_values(i) & plot_cluster_values == cluster_idx);
        num_points = length(indices);

        % Append the current group of x_positions
        x_positions{cluster_idx} = [x_positions{cluster_idx}, pos_counter:pos_counter+num_points-1];
        y_positions{cluster_idx} = [y_positions{cluster_idx}, psth_bin_RT_trial_split_mean_max_ampl(indices)'];
        x_labels_wf_max{cluster_idx} = [x_labels_wf_max{cluster_idx}; arrayfun(@(a, b) sprintf('(%d,%d)', a, b), ...
            plot_day_values(indices), plot_trial_values(indices), 'UniformOutput', false)];
        error_vals{cluster_idx} = [error_vals{cluster_idx}, psth_bin_RT_trial_split_sem_max_ampl(indices)'];

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
    imagesc(squeeze(centroid_images(cluster_idx,:,:)))
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    title(['Cluster ', num2str(cluster_idx)]);

    nexttile;
    errorbar(x_positions{cluster_idx}, y_positions{cluster_idx}, error_vals{cluster_idx}, '-o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
    xticks(valid_xticks{cluster_idx}); % Only set valid x-ticks
    xticklabels(valid_xticklabels{cluster_idx}); % Remove NaN labels
    xtickangle(45); % Rotate for better visibility
    ylim([0 8])
    title('PSTH max amplitude mean')
    sgtitle(['RT bin split Cluster ' num2str(cluster_idx)]);
end

%% consecutive - get 'good' RT trial sequences
good_RT_range = [0.01, 0.5];
good_RT_binary_cell = cell(size(all_RT));
good_RT_sequence_lengths_cell = cell(size(all_RT)); % Store sequence lengths per recording
for rec_idx = 1:num_recordings
    RT_values = all_RT{rec_idx};

    good_RT_trials = (RT_values >= good_RT_range(1) & RT_values <= good_RT_range(2));
    good_RT_binary_cell{rec_idx} = double(good_RT_trials);

    good_RT_sequence_lengths = zeros(size(good_RT_binary_cell{rec_idx}));
    seq_counter = 0;

    for i = 1:length(good_RT_binary_cell{rec_idx})
        if good_RT_binary_cell{rec_idx}(i) == 1
            seq_counter = seq_counter + 1;  
        else
            seq_counter = 0;  
        end
        good_RT_sequence_lengths(i) = seq_counter; 
    end

    good_RT_sequence_lengths_cell{rec_idx} = good_RT_sequence_lengths;
end
for_psth_good_RT_sequence_lengths_cell = arrayfun(@(rec_idx) ...
    repmat(good_RT_sequence_lengths_cell{rec_idx}', 1, psth_num_clusters(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_good_RT_sequence_lengths = vertcat(for_psth_good_RT_sequence_lengths_cell{:});

% combined good sequence 
for_psth_combined_good_RT_sequence_lengths = for_psth_good_RT_sequence_lengths;
for_psth_combined_good_RT_sequence_lengths(for_psth_good_RT_sequence_lengths>10) = 10;


%% - avg across animals in 'good' RT trial sequences groups
[avg_psth_grouped_combined_good_RT_sequence_smooth_norm,psth_unique_avg_animal_group_combined_good_RT_sequence_indices] = ...
    ap.nestgroupfun({@mean, @mean},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_combined_good_RT_sequence_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
num_animals_stim_psth_combined_good_RT_sequence = ...
    mean(ap.nestgroupfun({@mean, @length},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_combined_good_RT_sequence_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]), ...
    2);

%% get max ampl for good sequence
% per mouse
[psth_all_grouped_combined_good_RT_sequence_max_ampl, ...
    psth_unique_combined_good_RT_sequence_indices] = ap.nestgroupfun({@(x) x, @mean},...
    psth_all_trial_split_max_ampl, ...
    (1:length(for_psth_animal_ids))', ...
    [for_psth_animal_ids, for_psth_combined_good_RT_sequence_lengths, ...
    for_psth_combined_days_from_learning, for_psth_cluster_ids]);

[psth_all_grouped_combined_good_RT_sequence_n_trials] = ap.nestgroupfun({@(x) x, @length},...
    psth_all_trial_split_max_ampl, ...
    (1:length(for_psth_animal_ids))', ...
    [for_psth_animal_ids, for_psth_combined_good_RT_sequence_lengths, ...
    for_psth_combined_days_from_learning, for_psth_cluster_ids]);

% across mice
psth_combined_good_RT_sequence_mean_max_ampl = ap.nestgroupfun({@mean, @mean},...
    psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_combined_good_RT_sequence_lengths, ...
    for_psth_combined_days_from_learning, for_psth_cluster_ids]);

% do sem for errorbar
psth_combined_good_RT_sequence_std_max_ampl = ap.nestgroupfun({@mean, @nanstd},...
    psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_combined_good_RT_sequence_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
psth_combined_good_RT_sequence_sem_max_ampl = psth_combined_good_RT_sequence_std_max_ampl ./ sqrt(num_animals_stim_psth_combined_good_RT_sequence);

%% plot good RT sequence
curr_color = 'k';
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        this_day = psth_unique_combined_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_combined_good_RT_sequence_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_combined_good_RT_sequence_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_mean_max_ampl = psth_combined_good_RT_sequence_mean_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_sem_max_ampl = psth_combined_good_RT_sequence_sem_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_seq_length = psth_unique_avg_animal_group_combined_good_RT_sequence_indices(this_day_idx & this_cluster_idx, 1);
        
        nexttile;
%         plot(for_plot_mean_RT, for_plot_mean_max_ampl);

         errorbar(for_plot_seq_length, for_plot_mean_max_ampl, for_plot_sem_max_ampl, 'o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
         ylim([0 15]);
%         hold on;
%         xline(for_plot_mean_RT)
        yline(0)
        ylabel('Max ampl of str response')
        xlabel('Seq length in group')
        title(['Day ' num2str(psth_unique_combined_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['Str response vs seq length for Cluster ' num2str(cluster_idx)])
end

%% try swarm plot
curr_color = 'k';
gray_color = [0.7 0.7 0.7]; 

for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        this_day = psth_unique_combined_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_combined_good_RT_sequence_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_combined_good_RT_sequence_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_mean_max_ampl = psth_combined_good_RT_sequence_mean_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_sem_max_ampl = psth_combined_good_RT_sequence_sem_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_seq_length = psth_unique_avg_animal_group_combined_good_RT_sequence_indices(this_day_idx & this_cluster_idx, 1);

        for_swarm_idx = psth_unique_combined_good_RT_sequence_indices(:,3) == this_day & ...
                psth_unique_combined_good_RT_sequence_indices(:,4) == cluster_idx;
        individual_data = psth_all_grouped_combined_good_RT_sequence_max_ampl(for_swarm_idx);
        individual_seq_lengths = psth_unique_combined_good_RT_sequence_indices(for_swarm_idx, 2); 
        
        nexttile;
        swarmchart(individual_seq_lengths, test, ...
                   20, gray_color, 'filled', 'MarkerFaceAlpha', 0.5);
        hold on;
        errorbar(for_plot_seq_length, for_plot_mean_max_ampl, for_plot_sem_max_ampl, 'o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
         ylim([0 15]);
        yline(0)
        ylabel('Max ampl of str response')
        xlabel('Seq length in group')
        title(['Day ' num2str(psth_unique_combined_days_from_learning(day_idx))])
    end
    sgtitle(['Str response vs seq length for Cluster ' num2str(cluster_idx)])
end

%% norm each mouse to seq 0 on each day
psth_good_RT_sequence_max_ampl_norm = cell(num_clusters, ...
    length(psth_unique_combined_days_from_learning), ...
        length(unique_animal_ids));
psth_good_RT_seq_length_norm = cell(num_clusters, ...
    length(psth_unique_combined_days_from_learning), ...
        length(unique_animal_ids));
psth_min_trials_used = 5;
for animal_idx=1:length(unique_animal_ids)
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        for cluster_idx=1:num_clusters
            all_seq_idx = find(psth_unique_combined_good_RT_sequence_indices(:, 1) == animal_idx & ...
                psth_unique_combined_good_RT_sequence_indices(:, 3) == psth_unique_combined_days_from_learning(day_idx) & ...
                psth_unique_combined_good_RT_sequence_indices(:, 4) == cluster_idx);
            this_seq_max = psth_all_grouped_combined_good_RT_sequence_max_ampl(all_seq_idx);               ;
            this_baseline = psth_all_grouped_combined_good_RT_sequence_max_ampl...
                (psth_unique_combined_good_RT_sequence_indices(:, 1) == animal_idx & ...
                psth_unique_combined_good_RT_sequence_indices(:, 3) == psth_unique_combined_days_from_learning(day_idx) & ...
                psth_unique_combined_good_RT_sequence_indices(:, 4) == cluster_idx &...
                psth_unique_combined_good_RT_sequence_indices(:, 2) == 0);
            this_norm_seq_max = this_seq_max / this_baseline;
            this_norm_seq_max(...
                psth_all_grouped_combined_good_RT_sequence_n_trials...
                (all_seq_idx) < psth_min_trials_used) = NaN;
            psth_good_RT_sequence_max_ampl_norm{cluster_idx, day_idx, animal_idx} = this_norm_seq_max;
            psth_good_RT_seq_length_norm{cluster_idx, day_idx, animal_idx} = ...
                psth_unique_combined_good_RT_sequence_indices(all_seq_idx, 2);
        end
    end
end

% plot
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        this_good_RT_sequence_max_ampl_norm = {};
        nexttile;
        for animal_idx=1:length(unique_animal_ids)
            plot(psth_good_RT_seq_length_norm{cluster_idx, day_idx, animal_idx}, ...
                psth_good_RT_sequence_max_ampl_norm{cluster_idx, day_idx, animal_idx}, '-o');
            hold on;
        end
        ylim([0 4]);
        yline(0)
        ylabel('Max ampl of str response')
        xlabel('Seq length in group')
        title(['Day ' num2str(psth_unique_combined_days_from_learning(day_idx))])
    end
end

%% cumulative
good_RT_range = [0.01, 0.5];
good_RT_binary_cell = cell(size(all_RT));
good_RT_cumulative_lengths_cell = cell(size(all_RT)); 
for rec_idx = 1:num_recordings
    RT_values = all_RT{rec_idx};

    good_RT_trials = (RT_values >= good_RT_range(1) & RT_values <= good_RT_range(2));
    good_RT_binary_cell{rec_idx} = double(good_RT_trials);

    good_RT_cumulative_lengths = zeros(size(good_RT_binary_cell{rec_idx}));
    good_RT_cumulative_lengths(good_RT_trials) = cumsum(good_RT_binary_cell{rec_idx}(good_RT_trials));
    good_RT_cumulative_lengths_cell{rec_idx} = good_RT_cumulative_lengths;
end
for_psth_good_RT_cumulative_lengths_cell = arrayfun(@(rec_idx) ...
    repmat(good_RT_cumulative_lengths_cell{rec_idx}', 1,  psth_num_clusters(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_good_RT_cumulative_lengths = vertcat(for_psth_good_RT_cumulative_lengths_cell{:});

% combined good cumulative
combine_cumulative = 20;
for_psth_combined_good_RT_cumulative_lengths = for_psth_good_RT_cumulative_lengths;
for_psth_combined_good_RT_cumulative_lengths(for_psth_good_RT_cumulative_lengths>combine_cumulative) = combine_cumulative;

%% - avg across animals in cumulative RT trial sequences groups
[avg_psth_grouped_combined_good_RT_cumulative_smooth_norm,...
    psth_unique_avg_animal_group_combined_good_RT_cumulative_idx] = ...
    ap.nestgroupfun({@mean, @mean},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_combined_good_RT_cumulative_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
num_animals_stim_psth_combined_good_RT_cumulative = ...
    mean(ap.nestgroupfun({@mean, @length},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_combined_good_RT_cumulative_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]), ...
    2);

%% get max ampl for cumulative good sequence
% group by ld and split trial
psth_combined_good_RT_cumulative_mean_max_ampl = ap.nestgroupfun({@mean, @mean},psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_combined_good_RT_cumulative_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]);

% do sem for errorbar
psth_combined_good_RT_cumulative_std_max_ampl = ap.nestgroupfun({@mean, @nanstd},psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_combined_good_RT_cumulative_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
psth_combined_good_RT_cumulative_sem_max_ampl = psth_combined_good_RT_cumulative_std_max_ampl ./ sqrt(num_animals_stim_psth_combined_good_RT_cumulative);

%% plot cumulative
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        this_day = psth_unique_combined_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_combined_good_RT_cumulative_idx(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_combined_good_RT_cumulative_idx(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_mean_max_ampl = psth_combined_good_RT_cumulative_mean_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_sem_max_ampl = psth_combined_good_RT_cumulative_sem_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_seq_length = psth_unique_avg_animal_group_combined_good_RT_cumulative_idx(this_day_idx & this_cluster_idx, 1);
        
        nexttile;
%         plot(for_plot_mean_RT, for_plot_mean_max_ampl);

         errorbar(for_plot_seq_length, for_plot_mean_max_ampl, for_plot_sem_max_ampl, 'o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
       ylim([0 20]);
%         hold on;
%         xline(for_plot_mean_RT)
        yline(0)
        ylabel('Max ampl of str response')
        xlabel('Seq length in group')
        title(['Day ' num2str(psth_unique_combined_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['Str response vs cumulative for Cluster ' num2str(cluster_idx)])
end

%% trials back
num_trials_back = 10;
good_RT_range = [0.01, 0.5];
good_RT_trials_back_lengths_cell = cell(size(all_RT)); 
for rec_idx = 1:num_recordings
    RT_values = all_RT{rec_idx};

    good_RT_trials = (RT_values >= good_RT_range(1) & RT_values <= good_RT_range(2));
    good_RT_binary_cell{rec_idx} = double(good_RT_trials);

    good_RT_trials_back_lengths = nan(size(good_RT_binary_cell{rec_idx}));
    for i=1:length(good_RT_binary_cell{rec_idx})
        if i<=num_trials_back
%             good_RT_trials_back_lengths(i) = sum(good_RT_binary_cell{rec_idx}(1:i-1));
            continue;
        elseif good_RT_binary_cell{rec_idx}(i) == 1
            good_RT_trials_back_lengths(i) = sum(good_RT_binary_cell{rec_idx}(i-num_trials_back:i-1));
        end
    end

    good_RT_trials_back_lengths_cell{rec_idx} = good_RT_trials_back_lengths;
end
for_psth_good_RT_trials_back_lengths_cell = arrayfun(@(rec_idx) ...
    repmat(good_RT_trials_back_lengths_cell{rec_idx}', 1, psth_num_clusters(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_good_RT_trials_back_lengths = vertcat(for_psth_good_RT_trials_back_lengths_cell{:});

%% - avg across animals in trials_back RT trial sequences groups
[avg_psth_grouped_good_RT_trials_back_smooth_norm,psth_unique_avg_animal_group_good_RT_trials_back_indices] = ...
    ap.nestgroupfun({@mean, @mean},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_good_RT_trials_back_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
num_animals_stim_psth_good_RT_trials_back = ...
    mean(ap.nestgroupfun({@mean, @length},cat_smooth_norm_stim_psths, ...
    for_psth_animal_ids, ...
    [for_psth_good_RT_trials_back_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]), ...
    2);

%% get max ampl for trials_back good sequence
% group by ld and split trial
psth_good_RT_trials_back_mean_max_ampl = ap.nestgroupfun({@mean, @mean},psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_good_RT_trials_back_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]);

% do sem for errorbar
psth_good_RT_trials_back_std_max_ampl = ap.nestgroupfun({@mean, @nanstd},psth_all_trial_split_max_ampl, ...
    for_psth_animal_ids, ...
    [for_psth_good_RT_trials_back_lengths, for_psth_combined_days_from_learning, for_psth_cluster_ids]);
psth_good_RT_trials_back_sem_max_ampl = psth_good_RT_trials_back_std_max_ampl ./ sqrt(num_animals_stim_psth_good_RT_trials_back);

%% - plot trials_back
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_combined_days_from_learning)
        this_day = psth_unique_combined_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_good_RT_trials_back_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_good_RT_trials_back_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_mean_max_ampl = psth_good_RT_trials_back_mean_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_sem_max_ampl = psth_good_RT_trials_back_sem_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_seq_length = psth_unique_avg_animal_group_good_RT_trials_back_indices(this_day_idx & this_cluster_idx, 1);
        
        nexttile;
%         plot(for_plot_mean_RT, for_plot_mean_max_ampl);

         errorbar(for_plot_seq_length, for_plot_mean_max_ampl, for_plot_sem_max_ampl, 'o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
       ylim([0 25]);
%         hold on;
%         xline(for_plot_mean_RT)
        yline(0)
        ylabel('Max ampl of str response')
        xlabel('Seq length in group')
        title(['Day ' num2str(psth_unique_combined_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['Str response vs trials back for Cluster ' num2str(cluster_idx)])
end
