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
%% check the nan cluster ids

nan_cluster_ids = find(isnan(cluster_ids));
figure;
tiledlayout('flow')
for i=1:length(nan_cluster_ids)
    nexttile;
    imagesc(all_cortex_kernel_px(:,:,nan_cluster_ids(i)))
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
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
psth_num_trials = arrayfun(@(rec_idx) ...
    size(task_ephys.binned_spikes_stim_align{rec_idx}, 1) * ~isempty(task_ephys.binned_spikes_stim_align{rec_idx}), ...
    1:num_recordings);
for_psth_cluster_ids_cell = arrayfun(@(rec_idx) repelem(per_rec_psth_cluster_ids{rec_idx},psth_num_trials(rec_idx))', ...
    1:num_recordings, 'uni', false);
for_psth_cluster_ids = vertcat(for_psth_cluster_ids_cell{:});

% get trial ids
for_psth_trial_ids_cell = arrayfun(@(rec_idx) repmat(1:psth_num_trials(rec_idx), 1, n_depths(rec_idx))', ...
    1:num_recordings, 'uni', false);
for_psth_trial_ids = vertcat(for_psth_trial_ids_cell{:});

% disp(length(for_psth_cluster_ids_all))
% disp(sum(n_depths.*num_trials))

% get days from learning and animal id
for_psth_days_from_learning = repelem(bhv.days_from_learning, n_depths.*psth_num_trials);
[~, ~, for_psth_animal_ids] = unique(repelem(bhv.animal, n_depths.*psth_num_trials));

% get flattened psths
% permute first
permute_stim_binned_spikes = arrayfun(@(rec_idx) ...
    permute(task_ephys.binned_spikes_stim_align{rec_idx}, [1 3 2]), ...
    1:num_recordings, 'uni', false);
flattened_stim_binned_spikes = arrayfun(@(rec_idx) ...
    reshape(permute_stim_binned_spikes{rec_idx}, ...
    n_depths(rec_idx)*psth_num_trials(rec_idx), ...
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
psth_use_days = ~isnan(for_psth_days_from_learning);

[unique_cluster_ids, ~, ~] = unique(for_psth_cluster_ids(psth_use_days));
[psth_unique_days_from_learning, ~, ~] = unique(for_psth_days_from_learning(psth_use_days));
[unique_animal_ids, ~, ~] = unique(for_psth_animal_ids(psth_use_days));

% Combine cluster_ids and days into a single matrix for indexing
psth_group_indices = [for_psth_trial_ids(psth_use_days), for_psth_animal_ids(psth_use_days), for_psth_cluster_ids(psth_use_days), for_psth_days_from_learning(psth_use_days)];
[psth_group_indices_unique_clusters, ~, psth_group_clusters_indices] = unique(psth_group_indices, 'rows');

grouped_cluster_stim_psths = ap.groupfun(@sum, ...
    cat_flattened_stim_binned_spikes(psth_use_days, :), psth_group_clusters_indices, []);


%% norm and smooth
psth_stim_time = -0.5:0.001:1;
baseline_idx = psth_stim_time >= -0.2 & psth_stim_time <= 0;

% Function to normalize and smooth
gauss_win = gausswin(51, 3)';
normalize_and_smooth = @(psth) ...
    filter(gauss_win, sum(gauss_win), (psth - mean(psth(baseline_idx), 2)) ./ mean(psth(baseline_idx), 2), [], 2);

grouped_cluster_smooth_norm_stim_psths = cell2mat(arrayfun(@(rep) ...
    normalize_and_smooth(grouped_cluster_stim_psths(rep, :)), ...
    1:size(grouped_cluster_stim_psths, 1), ...
    'UniformOutput', false)');

%% package in cluster and day

norm_smooth_stim_psths = cell(num_clusters, length(psth_unique_days_from_learning));
for cluster_idx=1:num_clusters
    for day_idx=1:length(psth_unique_days_from_learning)
        norm_smooth_stim_psths{cluster_idx, day_idx} = ...
            grouped_cluster_smooth_norm_stim_psths(...
            psth_group_indices_unique_clusters(:, 3) == cluster_idx & ...
            psth_group_indices_unique_clusters(:, 4) == psth_unique_days_from_learning(day_idx), :);
    end
end


%% plot sorted by resp
sort_time = psth_stim_time > 0 & psth_stim_time < 0.2;
sorted_norm_smooth_stim_psths = cell(num_clusters, length(psth_unique_days_from_learning));
for cluster_idx=1:num_clusters
    for day_idx=1:length(psth_unique_days_from_learning)
        [~, sort_idx] = sort(mean(norm_smooth_stim_psths{cluster_idx, day_idx}(:, sort_time), 2), 'descend');
        sorted_norm_smooth_stim_psths{cluster_idx, day_idx} = norm_smooth_stim_psths{cluster_idx, day_idx}(sort_idx, :);
    end
end

plot_time = [-0.5 1];
plot_time_idx = psth_stim_time > plot_time(1) & psth_stim_time < plot_time(2);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow')
    for day_idx=1:length(psth_unique_days_from_learning)
        nexttile;
        imagesc(psth_stim_time(plot_time_idx), [], sorted_norm_smooth_stim_psths{cluster_idx, day_idx}(:, plot_time_idx))
        max_abs = max(abs(sorted_norm_smooth_stim_psths{cluster_idx, day_idx}(:, plot_time_idx)), [], 'all');
        if isempty(max_abs)
            continue
        end
        clim([-max_abs max_abs])
        colormap(ap.colormap('BWR'))
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
    end
    sgtitle(['Cluster ' num2str(cluster_idx)])
end

% test = group_indices_unique_clusters(:,[1,3]);
% size(unique(test, 'rows'))
%
% sum(cellfun(@(x) ~isempty(x), ephys.trial_stim_values) & ~isnan(bhv.days_from_learning))

%% make RT sorting
all_RT = vertcat(bhv.stim_to_move);
for_psth_RT_cell = arrayfun(@(rec_idx) repmat(all_RT{rec_idx},n_depths(rec_idx), 1), ...
    1:num_recordings, 'uni', false);
for_psth_RT = vertcat(for_psth_RT_cell{:});

grouped_cluster_RT = ap.groupfun(@mean, ...
    for_psth_RT(psth_use_days), psth_group_clusters_indices, []);

RT_cluster_day = cell(num_clusters, length(psth_unique_days_from_learning));
RT_sort_idx = cell(num_clusters, length(psth_unique_days_from_learning));
for cluster_idx=1:num_clusters
    for day_idx=1:length(psth_unique_days_from_learning)
        RT_cluster_day{cluster_idx, day_idx} = ...
            grouped_cluster_RT(...
            psth_group_indices_unique_clusters(:, 3) == cluster_idx & ...
            psth_group_indices_unique_clusters(:, 4) == psth_unique_days_from_learning(day_idx), :);
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
        plot(RT_cluster_day{cluster_idx, day_idx}(RT_sort_idx{cluster_idx, day_idx}),1:size(RT_cluster_day{cluster_idx, day_idx}, 1))
        if isempty(max_abs)
            continue
        end
        %         clim([-max_abs max_abs])
        clim([-2,2]);
        colormap(ap.colormap('BWR'))
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
    end
    sgtitle(['Cluster ' num2str(cluster_idx)])
end

%% - split day into thirds

%% - split trials into 3

num_split_trials = 3;

% for_psth_trial_ids_cell = arrayfun(@(rec_idx) repmat(1:psth_num_trials(rec_idx), 1, n_depths(rec_idx))', ...
%     1:num_recordings, 'uni', false);
% for_psth_trial_ids = vertcat(for_psth_trial_ids_cell{:});

temp_for_psth_split_trial_ids_cell = arrayfun(@(rec_idx) repelem(1:num_split_trials, ceil(psth_num_trials(rec_idx) / num_split_trials)), ...
    1:num_recordings, 'UniformOutput', false);
temp_for_psth_split_trial_ids_cell_trimmed = arrayfun(@(rec_idx) temp_for_psth_split_trial_ids_cell{rec_idx}(1:psth_num_trials(rec_idx)), ...
    1:num_recordings, 'UniformOutput', false);
for_psth_split_trial_ids_cell = arrayfun(@(rec_idx) repmat(temp_for_psth_split_trial_ids_cell_trimmed{rec_idx}, 1, n_depths(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_split_trial_ids = vertcat(for_psth_split_trial_ids_cell{:});

%% get split trial indices
psth_split_trial_indices = [for_psth_animal_ids(psth_use_days), for_psth_split_trial_ids(psth_use_days), ...
    for_psth_days_from_learning(psth_use_days), for_psth_cluster_ids(psth_use_days)];
[psth_unique_split_trial_indices, ~, psth_split_trial_indices] = unique(psth_split_trial_indices, 'rows');

psth_grouped_cluster_split_trial = ap.groupfun(@sum, ...
    cat_flattened_stim_binned_spikes(psth_use_days, :), psth_split_trial_indices, []);

%% norm and smooth split trial
psth_stim_time = -0.5:0.001:1;
baseline_idx = psth_stim_time >= -0.2 & psth_stim_time <= 0;

% Function to normalize and smooth
gauss_win = gausswin(51, 3)';
normalize_and_smooth = @(psth) ...
    filter(gauss_win, sum(gauss_win), (psth - mean(psth(baseline_idx), 2)) ./ mean(psth(baseline_idx), 2), [], 2);

psth_grouped_cluster_split_trial_smooth_norm = cell2mat(arrayfun(@(rep) ...
    normalize_and_smooth(psth_grouped_cluster_split_trial(rep, :)), ...
    1:size(psth_grouped_cluster_split_trial, 1), ...
    'UniformOutput', false)');

%% group by ld and split trial
psth_day_trial_indices = [psth_unique_split_trial_indices(:, 2), psth_unique_split_trial_indices(:, 3), ...
    psth_unique_split_trial_indices(:, 4)];
[psth_unique_avg_animal_group_indices, ~, psth_animal_group_clusters_indices] = unique(psth_day_trial_indices, 'rows');

avg_psth_grouped_split_trial_smooth_norm = ap.groupfun(@mean, ...
    psth_grouped_cluster_split_trial_smooth_norm, psth_animal_group_clusters_indices, []);

%% count animals
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


%% get max ampl
psth_window_for_max = psth_stim_time>0 & psth_stim_time<0.3;
psth_all_grouped_trial_split_max_ampl = max(psth_grouped_cluster_split_trial_smooth_norm(:, psth_window_for_max), [], 2);

% group by ld and split trial
psth_trial_split_mean_max_ampl = ap.groupfun(@mean, ...
    psth_all_grouped_trial_split_max_ampl, psth_animal_group_clusters_indices);

% do sem for errorbar
psth_trial_split_std_max_ampl = ap.groupfun(@nanstd, ...
    psth_all_grouped_trial_split_max_ampl, psth_animal_group_clusters_indices);
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

%% split into 3 by RT sorting
num_RT_split_trials = 5;
temp_for_psth_sort_RT_split_trial_ids_cell = cell(1, num_recordings);
for rec_idx=1:num_recordings
    % (to discretize by RT rank)
    rank_RT_per_rec = tiedrank(all_RT{rec_idx});
    temp_for_psth_sort_RT_split_trial_ids_cell{rec_idx} = discretize(rank_RT_per_rec,...
        round(linspace(1,max(rank_RT_per_rec),num_RT_split_trials+1)))';
end
for_psth_sort_RT_split_trial_ids_cell = arrayfun(@(rec_idx) repmat(temp_for_psth_sort_RT_split_trial_ids_cell{rec_idx}, 1, n_depths(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_sort_RT_split_trial_ids = vertcat(for_psth_sort_RT_split_trial_ids_cell{:});

% get sorted RT
psth_per_rec_RT_cell = arrayfun(@(rec_idx) repmat(all_RT{rec_idx}', 1, n_depths(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
psth_per_rec_RT = vertcat(psth_per_rec_RT_cell{:});

%% make sort RT split trial indices
psth_sort_RT_split_trial_group_indices = [for_psth_animal_ids(psth_use_days), for_psth_sort_RT_split_trial_ids(psth_use_days), ...
    for_psth_days_from_learning(psth_use_days), for_psth_cluster_ids(psth_use_days)];
[psth_unique_sort_RT_split_trial_indices, ~, psth_sort_RT_split_trial_indices] = unique(psth_sort_RT_split_trial_group_indices, 'rows');

psth_grouped_cluster_sort_RT_split_trial = ap.groupfun(@sum, ...
    cat_flattened_stim_binned_spikes(psth_use_days, :), psth_sort_RT_split_trial_indices, []);

psth_sort_RT_split_trial_RT_mean = ap.groupfun(@mean, ...
    psth_per_rec_RT(psth_use_days), psth_sort_RT_split_trial_indices);

%% norm and smooth sort RT split trial
% psth_stim_time = -0.5:0.001:1;
% baseline_idx = psth_stim_time >= -0.2 & psth_stim_time <= 0;
%
% % Function to normalize and smooth
% gauss_win = gausswin(51, 3)';
% normalize_and_smooth = @(psth) ...
%     filter(gauss_win, sum(gauss_win), (psth - mean(psth(baseline_idx), 2)) ./ mean(psth(baseline_idx), 2), [], 2);

psth_grouped_cluster_sort_RT_split_trial_smooth_norm = cell2mat(arrayfun(@(rep) ...
    normalize_and_smooth(psth_grouped_cluster_sort_RT_split_trial(rep, :)), ...
    1:size(psth_grouped_cluster_sort_RT_split_trial, 1), ...
    'UniformOutput', false)');

%% group by ld and RT sort split trial
psth_sort_RT_day_trial_indices = [psth_unique_sort_RT_split_trial_indices(:, 2), psth_unique_sort_RT_split_trial_indices(:, 3), ...
    psth_unique_sort_RT_split_trial_indices(:, 4)];
[psth_unique_avg_animal_group_sort_RT_indices, ~, psth_animal_group_clusters_sort_RT_indices] = unique(psth_sort_RT_day_trial_indices, 'rows');

avg_psth_grouped_sort_RT_split_trial_smooth_norm = ap.groupfun(@mean, ...
    psth_grouped_cluster_sort_RT_split_trial_smooth_norm, psth_animal_group_clusters_sort_RT_indices, []);

% count animals
num_animals_stim_psth_sort_RT_split = accumarray(psth_animal_group_clusters_sort_RT_indices, 1);

% get mean RT
avg_psth_sort_RT_split_trial_RT_mean = ap.groupfun(@mean, ...
    psth_sort_RT_split_trial_RT_mean, psth_animal_group_clusters_sort_RT_indices);

%% plot RT sort split trial psths
my_colormap = ap.colormap('KR', num_RT_split_trials);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = psth_unique_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_sort_RT_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_sort_RT_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_psth = avg_psth_grouped_sort_RT_split_trial_smooth_norm(this_day_idx & this_cluster_idx, :);
        for_plot_mean_RT = avg_psth_sort_RT_split_trial_RT_mean(this_day_idx & this_cluster_idx);
        nexttile;
        plot(psth_stim_time, for_plot_psth);
        ylim([-0.5 4]);
        xlim([-0.5 1]);
        hold on;
        xline(for_plot_mean_RT)
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['RT sort split Cluster ' num2str(cluster_idx)])
end

%% max ampl for sort RT split
%% get max ampl sort RT
psth_window_for_max = psth_stim_time>0 & psth_stim_time<0.3;
psth_all_grouped_sort_RT_trial_split_max_ampl = max(psth_grouped_cluster_sort_RT_split_trial_smooth_norm(:, psth_window_for_max), [], 2);

% group by ld and split trial
psth_sort_RT_trial_split_mean_max_ampl = ap.groupfun(@mean, ...
    psth_all_grouped_sort_RT_trial_split_max_ampl, psth_animal_group_clusters_sort_RT_indices);

psth_sort_RT_trial_split_median_max_ampl = ap.groupfun(@median, ...
    psth_all_grouped_sort_RT_trial_split_max_ampl, psth_animal_group_clusters_sort_RT_indices);

% do sem for errorbar
psth_sort_RT_trial_split_std_max_ampl = ap.groupfun(@nanstd, ...
    psth_all_grouped_sort_RT_trial_split_max_ampl, psth_animal_group_clusters_sort_RT_indices);
psth_sort_RT_trial_split_sem_max_ampl = psth_sort_RT_trial_split_std_max_ampl ./ sqrt(num_animals_stim_psth_sort_RT_split);

%% plot max ampl sort RT

curr_color = 'k';

plot_day_values = psth_unique_avg_animal_group_sort_RT_indices(:,2); % Extract x values
unique_plot_day_values = unique(plot_day_values);
plot_trial_values = psth_unique_avg_animal_group_sort_RT_indices(:,1); % Extract y values
plot_cluster_values = psth_unique_avg_animal_group_sort_RT_indices(:,3);

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
    title('PSTH max amplitude mean')
    sgtitle(['RT sort Cluster ' num2str(cluster_idx)]);
end


%% sort RT vs str response per day 
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = psth_unique_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_sort_RT_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_sort_RT_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_mean_max_ampl = psth_sort_RT_trial_split_mean_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_sem_max_ampl = psth_sort_RT_trial_split_sem_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_mean_RT = avg_psth_sort_RT_split_trial_RT_mean(this_day_idx & this_cluster_idx);
        nexttile;
%         plot(for_plot_mean_RT, for_plot_mean_max_ampl);

         errorbar(for_plot_mean_RT, for_plot_mean_max_ampl, for_plot_sem_max_ampl, 'o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
%         ylim([-0.5 4]);
%         hold on;
%         xline(for_plot_mean_RT)
        yline(0)
        ylabel('Max ampl of str response')
        xlabel('Mean RT in group')
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['RT sort vs str response for Cluster ' num2str(cluster_idx)])
end

%% combine mean on one plot
days_for_plot = -3:2;
all_colormap = ap.colormap('BKR', 2*max(abs(days_for_plot))+1);
colormap_days = -max(abs(days_for_plot)):max(abs(days_for_plot));
plot_day_idx = ismember(psth_unique_days_from_learning, days_for_plot);
% get right colours
plotted_days = psth_unique_days_from_learning(plot_day_idx);
my_colormap = all_colormap(ismember(colormap_days, plotted_days), :);

for cluster_idx=1:num_clusters
        figure;

    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = psth_unique_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_sort_RT_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_sort_RT_indices(:,3) == cluster_idx;
        if (sum(this_day_idx & this_cluster_idx) == 0) || (~ismember(this_day, days_for_plot))
            continue
        end
        for_plot_mean_max_ampl = psth_sort_RT_trial_split_mean_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_sem_max_ampl = psth_sort_RT_trial_split_sem_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_mean_RT = avg_psth_sort_RT_split_trial_RT_mean(this_day_idx & this_cluster_idx);

        % find curr color
        curr_color = my_colormap(ismember(plotted_days, this_day), :);

         errorbar(for_plot_mean_RT, for_plot_mean_max_ampl, for_plot_sem_max_ampl, '-o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
                 hold on;

%         ylim([-0.5 4]);
%         hold on;
%         xline(for_plot_mean_RT)

        yline(0)
        xlim([-0.5 5])
        ylabel('Max ampl of str response')
        xlabel('Mean RT in group')
    end
    sgtitle(['RT sort vs str response for Cluster ' num2str(cluster_idx)])
end

%% plot both split trial and sort RT split trial - ADD passive here
% days_for_plot = -3:2;
% % these_days_from_learning = psth_unique_days_from_learning;
% % plot_day_idx = ismember(these_days_from_learning, days_for_plot);
% % plotted_days = these_days_from_learning(plot_day_idx);
% RT_trial_split_colormap = ap.colormap('KR', num_RT_split_trials);
% trial_split_colormap = ap.colormap('KR', num_split_trials);
% for cluster_idx=1:num_clusters
%     figure;
%     tiledlayout(length(days_for_plot), 2);
%     for day_idx=1:length(psth_unique_days_from_learning)
%         this_day = psth_unique_days_from_learning(day_idx);
%         if ismember(this_day, days_for_plot)
%             this_day_idx = psth_unique_avg_animal_group_sort_RT_indices(:,2) == this_day;
%             this_cluster_idx =  psth_unique_avg_animal_group_sort_RT_indices(:,3) == cluster_idx;
%             if sum(this_day_idx & this_cluster_idx) == 0
%                 continue
%             end
% 
%             for_plot_RT_psth = avg_psth_grouped_sort_RT_split_trial_smooth_norm(this_day_idx & this_cluster_idx, :);
%             for_plot_mean_RT = avg_psth_RT_split_trial_RT_mean(this_day_idx & this_cluster_idx);
%             nexttile;
%             plot(psth_stim_time, for_plot_RT_psth);
%             ylim([-0.5 4]);
%             hold on;
%             xline(for_plot_mean_RT)
%             title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
%             colororder(gca, RT_trial_split_colormap);
% 
%             this_day_idx = psth_unique_avg_animal_group_indices(:,2) == this_day;
%             this_cluster_idx =  psth_unique_avg_animal_group_indices(:,3) == cluster_idx;
%             for_plot_psth = avg_psth_grouped_split_trial_smooth_norm(this_day_idx & this_cluster_idx, :);
%             nexttile;
%             plot(psth_stim_time, for_plot_psth);
%             ylim([-0.5 4]);
%             title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
%             colororder(gca, trial_split_colormap);
%         end
%     end
%     sgtitle(['RT split (left) and trial split (right) Cluster ' num2str(cluster_idx)])
% end

%% RT bin split - log rn
discretize_all_RT_bin = [-Inf,0:0.2:0.6,Inf];
log_discretize_all_RT_bin = [-Inf logspace(-1,1,5) Inf];
temp_for_psth_bin_RT_split_trial_ids_cell = cell(1, num_recordings);
for rec_idx=1:num_recordings
    % (to discretize by RT bin)
    temp_for_psth_bin_RT_split_trial_ids_cell{rec_idx} = discretize(all_RT{rec_idx},discretize_all_RT_bin)';
end
for_psth_bin_RT_split_trial_ids_cell = arrayfun(@(rec_idx) repmat(temp_for_psth_bin_RT_split_trial_ids_cell{rec_idx}, 1, n_depths(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_bin_RT_split_trial_ids = vertcat(for_psth_bin_RT_split_trial_ids_cell{:});

%% make bin RT split trial indices
psth_bin_RT_split_trial_group_indices = [for_psth_animal_ids(psth_use_days), for_psth_bin_RT_split_trial_ids(psth_use_days), ...
    for_psth_days_from_learning(psth_use_days), for_psth_cluster_ids(psth_use_days)];
[psth_unique_bin_RT_split_trial_indices, ~, psth_bin_RT_split_trial_indices] = unique(psth_bin_RT_split_trial_group_indices, 'rows');

psth_grouped_cluster_bin_RT_split_trial = ap.groupfun(@sum, ...
    cat_flattened_stim_binned_spikes(psth_use_days, :), psth_bin_RT_split_trial_indices, []);

psth_bin_RT_split_trial_RT_mean = ap.groupfun(@mean, ...
    psth_per_rec_RT(psth_use_days), psth_bin_RT_split_trial_indices);

%% norm and smooth bin RT split trial
% psth_stim_time = -0.5:0.001:1;
% baseline_idx = psth_stim_time >= -0.2 & psth_stim_time <= 0;
%
% % Function to normalize and smooth
% gauss_win = gausswin(51, 3)';
% normalize_and_smooth = @(psth) ...
%     filter(gauss_win, sum(gauss_win), (psth - mean(psth(baseline_idx), 2)) ./ mean(psth(baseline_idx), 2), [], 2);

psth_grouped_cluster_bin_RT_split_trial_smooth_norm = cell2mat(arrayfun(@(rep) ...
    normalize_and_smooth(psth_grouped_cluster_bin_RT_split_trial(rep, :)), ...
    1:size(psth_grouped_cluster_bin_RT_split_trial, 1), ...
    'UniformOutput', false)');

%% group by ld and RT bin split trial
psth_bin_RT_day_trial_indices = [psth_unique_bin_RT_split_trial_indices(:, 2), psth_unique_bin_RT_split_trial_indices(:, 3), ...
    psth_unique_bin_RT_split_trial_indices(:, 4)];
[psth_unique_avg_animal_group_bin_RT_indices, ~, psth_animal_group_clusters_bin_RT_indices] = unique(psth_bin_RT_day_trial_indices, 'rows');

avg_psth_grouped_bin_RT_split_trial_smooth_norm = ap.groupfun(@mean, ...
    psth_grouped_cluster_bin_RT_split_trial_smooth_norm, psth_animal_group_clusters_bin_RT_indices, []);

% count animals
num_animals_stim_psth_bin_RT_split = accumarray(psth_animal_group_clusters_bin_RT_indices, 1);

% get mean RT
avg_psth_bin_RT_split_trial_RT_mean = ap.groupfun(@mean, ...
    psth_bin_RT_split_trial_RT_mean, psth_animal_group_clusters_bin_RT_indices);

%% plot RT bin split trial psths
my_colormap = ap.colormap('KR', num_RT_split_trials);
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = psth_unique_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_bin_RT_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_bin_RT_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_psth = avg_psth_grouped_bin_RT_split_trial_smooth_norm(this_day_idx & this_cluster_idx, :);
        for_plot_mean_RT = avg_psth_bin_RT_split_trial_RT_mean(this_day_idx & this_cluster_idx);
        nexttile;
        plot(psth_stim_time, for_plot_psth);
        ylim([-0.5 4]);
        xlim([-0.5 1])
        hold on;
        xline(for_plot_mean_RT)
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['RT bin split Cluster ' num2str(cluster_idx)])
end

%% max ampl for bin RT split
%% get max ampl bin RT
psth_window_for_max = psth_stim_time>0 & psth_stim_time<0.3;
psth_all_grouped_bin_RT_trial_split_max_ampl = max(psth_grouped_cluster_bin_RT_split_trial_smooth_norm(:, psth_window_for_max), [], 2);

% group by ld and split trial
psth_bin_RT_trial_split_mean_max_ampl = ap.groupfun(@nanmean, ...
    psth_all_grouped_bin_RT_trial_split_max_ampl, psth_animal_group_clusters_bin_RT_indices);

% do sem for errorbar
psth_bin_RT_trial_split_std_max_ampl = ap.groupfun(@nanstd, ...
    psth_all_grouped_bin_RT_trial_split_max_ampl, psth_animal_group_clusters_bin_RT_indices);
psth_bin_RT_trial_split_sem_max_ampl = psth_bin_RT_trial_split_std_max_ampl ./ sqrt(num_animals_stim_psth_bin_RT_split);

%% plot max ampl bin RT

curr_color = 'k';

plot_day_values = psth_unique_avg_animal_group_bin_RT_indices(:,2); % Extract x values
unique_plot_day_values = unique(plot_day_values);
plot_trial_values = psth_unique_avg_animal_group_bin_RT_indices(:,1); % Extract y values
plot_cluster_values = psth_unique_avg_animal_group_bin_RT_indices(:,3);

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
    title('PSTH max amplitude mean')
    sgtitle(['RT Bin for Cluster ' num2str(cluster_idx)]);
end

%% RT bin vs str response per day 
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = psth_unique_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_bin_RT_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_bin_RT_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_mean_max_ampl = psth_bin_RT_trial_split_mean_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_sem_max_ampl = psth_bin_RT_trial_split_sem_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_mean_RT = avg_psth_bin_RT_split_trial_RT_mean(this_day_idx & this_cluster_idx);
        nexttile;
%         plot(for_plot_mean_RT, for_plot_mean_max_ampl);

         errorbar(for_plot_mean_RT, for_plot_mean_max_ampl, for_plot_sem_max_ampl, 'o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
%         ylim([-0.5 4]);
%         hold on;
%         xline(for_plot_mean_RT)
        yline(0)
        ylabel('Max ampl of str response')
        xlabel('Mean RT in group')
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['RT bin vs str response for Cluster ' num2str(cluster_idx)])
end

%% consecutive - get 'good' RT trial sequences
% good_RT_range = [0.01, 0.25];
% good_RT_binary_cell = cell(size(all_RT));
% 
% for rec_idx = 1:num_recordings
%     RT_values = all_RT{rec_idx};
%     good_RT_trials = (RT_values >= good_RT_range(1) & RT_values <= good_RT_range(2));
%     good_RT_binary_cell{rec_idx} = double(good_RT_trials);
% end
% good_RT_binary = vertcat(good_RT_binary_cell{:});


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
for_psth_good_RT_sequence_lengths_cell = arrayfun(@(rec_idx) repmat(good_RT_sequence_lengths_cell{rec_idx}', 1, n_depths(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_good_RT_sequence_lengths = vertcat(for_psth_good_RT_sequence_lengths_cell{:});


% 
% start_indices_cell = cellfun(@(x) find(diff([0; x]) == 1), good_RT_binary_cell, 'UniformOutput', false); 
% end_indices_cell = cellfun(@(x) find(diff([0; x]) == -1), good_RT_binary_cell, 'UniformOutput', false); 
% zeros_indices_cell = cellfun(@(x) find(x == 0), good_RT_binary_cell, 'UniformOutput', false); 
% cell_lengths_mismatch = cellfun(@(start_idx,end_idx) length(start_idx) ~= length(end_idx), start_indices_cell, end_indices_cell);
% % append an index for the end of the day so the sizes match 
% end_indices_cell(cell_lengths_mismatch) = cellfun(@(x, end_idx) [end_idx; length(x)], ...
%     good_RT_binary_cell(cell_lengths_mismatch), ...
%     end_indices_cell(cell_lengths_mismatch), 'UniformOutput', false);
% % get sequence lengths
% seq_lengths = cellfun(@(start_idx,end_idx) end_idx - start_idx, ...
%     start_indices_cell, end_indices_cell, 'UniformOutput', false);
% % make start to end idx cell
% good_RT_sequence_lengths = zeros(size(good_RT_binary));
% good_RT_seq = cellfun(@(start_idx, end_idx) arrayfun(@(s, e) 1:(e - s), start_idx, end_idx, 'UniformOutput', false), ...
%     start_indices_cell, end_indices_cell, 'UniformOutput', false);
% 
% sequences = arrayfun(@(s, e) 1:(e - s), start_idx, end_idx, 'UniformOutput', false);
% 
% start_idx = start_indices_cell{end};
% end_idx = end_indices_cell{end};
% zero_idx = zeros_indices_cell{end};
% this_seq_length = seq_lengths{end};
% 
% test1 = arrayfun(@(s, e) 1:(e - s), start_idx, end_idx, 'UniformOutput', false);
% 
% test_good = nan(1, psth_num_trials(end));
% test_good(start_idx) = 1;
% test_good(end_idx) = this_seq_length;
% 
% 
% 
% test_good_cell{start_idx} = test1;
% 
% 
% (start_idx:end_idx)
% 
% start_idx
% 
% test2 = horzcat(test1{:});
% figure;
% plot(test2, 'o')
% hold on;
% plot(good_RT_sequence_lengths_cell{end}, 'o')
% 
% 
% % find 
% one_indices = find(good_RT_binary == 1);
% % Identify where sequences reset (1 → 0 transition)
% start_indices = find(diff([0; good_RT_binary]) == 1); % Start of each sequence
% % get end indices
% end_indices = find(diff([0; good_RT_binary]) == -1);
% 
% length(start_indices) == length(end_indices)
% 
% good_RT_sequence_lengths = zeros(size(good_RT_binary));
% good_RT_sequence_lengths(one_indices) = interp1(one_indices, seq_values, one_indices, 'previous');

%% make indices for good RT sequence
% cluster id has nans!!!!!!!!!!!!
psth_good_RT_sequence_group_indices = [for_psth_animal_ids(psth_use_days), for_psth_good_RT_sequence_lengths(psth_use_days), ...
    for_psth_days_from_learning(psth_use_days), for_psth_cluster_ids(psth_use_days)];
[psth_unique_good_RT_sequence_indices, ~, psth_good_RT_sequence_indices] = unique(psth_good_RT_sequence_group_indices, 'rows');

psth_grouped_cluster_good_RT_sequence = ap.groupfun(@sum, ...
    cat_flattened_stim_binned_spikes(psth_use_days, :), psth_good_RT_sequence_indices, []);

%% norm and smooth good RT sequence
% psth_stim_time = -0.5:0.001:1;
% baseline_idx = psth_stim_time >= -0.2 & psth_stim_time <= 0;
%
% % Function to normalize and smooth
% gauss_win = gausswin(51, 3)';
% normalize_and_smooth = @(psth) ...
%     filter(gauss_win, sum(gauss_win), (psth - mean(psth(baseline_idx), 2)) ./ mean(psth(baseline_idx), 2), [], 2);

psth_grouped_cluster_good_RT_sequence_smooth_norm = cell2mat(arrayfun(@(rep) ...
    normalize_and_smooth(psth_grouped_cluster_good_RT_sequence(rep, :)), ...
    1:size(psth_grouped_cluster_good_RT_sequence, 1), ...
    'UniformOutput', false)');

%% group by ld and RT good RT sequence
psth_good_RT_sequence_day_trial_indices = [psth_unique_good_RT_sequence_indices(:, 2), ...
    psth_unique_good_RT_sequence_indices(:, 3), ...
    psth_unique_good_RT_sequence_indices(:, 4)];
[psth_unique_avg_animal_group_good_RT_sequence_indices, ~, ...
    psth_animal_group_clusters_good_RT_sequence_indices] = ...
    unique(psth_good_RT_sequence_day_trial_indices, 'rows');

avg_psth_grouped_good_RT_sequence_smooth_norm = ap.groupfun(@mean, ...
    psth_grouped_cluster_good_RT_sequence_smooth_norm, ....
    psth_animal_group_clusters_good_RT_sequence_indices, []);

% count animals
num_animals_stim_psth_good_RT_sequence = accumarray(psth_animal_group_clusters_good_RT_sequence_indices, 1);

%% get max ampl for good sequence
psth_window_for_max = psth_stim_time>0 & psth_stim_time<0.3;
psth_all_grouped_good_RT_sequence_max_ampl = max(psth_grouped_cluster_good_RT_sequence_smooth_norm(:, psth_window_for_max), [], 2);

% group by ld and seq length
psth_good_RT_sequence_mean_max_ampl = ap.groupfun(@mean, ...
    psth_all_grouped_good_RT_sequence_max_ampl, psth_animal_group_clusters_good_RT_sequence_indices);

% do sem for errorbar
psth_good_RT_sequence_std_max_ampl = ap.groupfun(@nanstd, ...
    psth_all_grouped_good_RT_sequence_max_ampl, psth_animal_group_clusters_good_RT_sequence_indices);
psth_good_RT_sequence_sem_max_ampl = psth_good_RT_sequence_std_max_ampl ./ sqrt(num_animals_stim_psth_good_RT_sequence);

%% plot good RT sequence
curr_color = 'k';
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = psth_unique_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_good_RT_sequence_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_good_RT_sequence_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_mean_max_ampl = psth_good_RT_sequence_mean_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_sem_max_ampl = psth_good_RT_sequence_sem_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_seq_length = psth_unique_avg_animal_group_good_RT_sequence_indices(this_day_idx & this_cluster_idx, 1);
        
        nexttile;
%         plot(for_plot_mean_RT, for_plot_mean_max_ampl);

         errorbar(for_plot_seq_length, for_plot_mean_max_ampl, for_plot_sem_max_ampl, 'o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
%         ylim([-0.5 4]);
%         hold on;
%         xline(for_plot_mean_RT)
        yline(0)
        ylabel('Max ampl of str response')
        xlabel('Seq length in group')
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
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
    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = psth_unique_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_good_RT_sequence_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_good_RT_sequence_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_mean_max_ampl = psth_good_RT_sequence_mean_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_sem_max_ampl = psth_good_RT_sequence_sem_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_seq_length = psth_unique_avg_animal_group_good_RT_sequence_indices(this_day_idx & this_cluster_idx, 1);
         

        for_swarm_idx = psth_unique_good_RT_sequence_indices(:,3) == this_day & ...
                psth_unique_good_RT_sequence_indices(:,4) == cluster_idx;
        individual_data = psth_all_grouped_good_RT_sequence_max_ampl(for_swarm_idx);
        individual_seq_lengths = psth_unique_good_RT_sequence_indices(for_swarm_idx, 2); 

        nexttile;
        swarmchart(individual_seq_lengths, individual_data, ...
                   20, gray_color, 'filled', 'MarkerFaceAlpha', 0.5);
        hold on;
        errorbar(for_plot_seq_length, for_plot_mean_max_ampl, for_plot_sem_max_ampl, 'o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
         ylim([0 15]);
%         hold on;
%         xline(for_plot_mean_RT)
        yline(0)
        ylabel('Max ampl of str response')
        xlabel('Seq length in group')
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
    end
    sgtitle(['Str response vs seq length for Cluster ' num2str(cluster_idx)])
end

%% cumulative
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
    repmat(good_RT_cumulative_lengths_cell{rec_idx}', 1, n_depths(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_good_RT_cumulative_lengths = vertcat(for_psth_good_RT_cumulative_lengths_cell{:});

%% make indices for cumulative good RT sequence
% cluster id has nans!!!!!!!!!!!!
psth_good_RT_cumulative_group_indices = [for_psth_animal_ids(psth_use_days), for_psth_good_RT_cumulative_lengths(psth_use_days), ...
    for_psth_days_from_learning(psth_use_days), for_psth_cluster_ids(psth_use_days)];
[psth_unique_good_RT_cumulative_indices, ~, psth_good_RT_cumulative_indices] = unique(psth_good_RT_cumulative_group_indices, 'rows');

psth_grouped_cluster_good_RT_cumulative = ap.groupfun(@sum, ...
    cat_flattened_stim_binned_spikes(psth_use_days, :), psth_good_RT_cumulative_indices, []);

%% norm and smooth sort RT split trial
psth_grouped_cluster_good_RT_cumulative_smooth_norm = cell2mat(arrayfun(@(rep) ...
    normalize_and_smooth(psth_grouped_cluster_good_RT_cumulative(rep, :)), ...
    1:size(psth_grouped_cluster_good_RT_cumulative, 1), ...
    'UniformOutput', false)');

%% group by ld and RT cumulative split trial
psth_good_RT_cumulative_day_trial_indices = [psth_unique_good_RT_cumulative_indices(:, 2), ...
    psth_unique_good_RT_cumulative_indices(:, 3), ...
    psth_unique_good_RT_cumulative_indices(:, 4)];
[psth_unique_avg_animal_group_good_RT_cumulative_indices, ~, ...
    psth_animal_group_clusters_good_RT_cumulative_indices] = ...
    unique(psth_good_RT_cumulative_day_trial_indices, 'rows');

avg_psth_grouped_good_RT_cumulative_smooth_norm = ap.groupfun(@mean, ...
    psth_grouped_cluster_good_RT_cumulative_smooth_norm, ....
    psth_animal_group_clusters_good_RT_cumulative_indices, []);

% count animals
num_animals_stim_psth_good_RT_cumulative = accumarray(psth_animal_group_clusters_good_RT_cumulative_indices, 1);

%% get max ampl for good cumulative
psth_window_for_max = psth_stim_time>0 & psth_stim_time<0.3;
psth_all_grouped_good_RT_cumulative_max_ampl = max(psth_grouped_cluster_good_RT_cumulative_smooth_norm(:, psth_window_for_max), [], 2);

% group by ld and seq length
psth_good_RT_cumulative_mean_max_ampl = ap.groupfun(@nanmean, ...
    psth_all_grouped_good_RT_cumulative_max_ampl, psth_animal_group_clusters_good_RT_cumulative_indices);

% do sem for errorbar
psth_good_RT_cumulative_std_max_ampl = ap.groupfun(@nanstd, ...
    psth_all_grouped_good_RT_cumulative_max_ampl, psth_animal_group_clusters_good_RT_cumulative_indices);
psth_good_RT_cumulative_sem_max_ampl = psth_good_RT_cumulative_std_max_ampl ./ sqrt(num_animals_stim_psth_good_RT_cumulative);

%% plot cumulative
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = psth_unique_days_from_learning(day_idx);
        this_day_idx = psth_unique_avg_animal_group_good_RT_cumulative_indices(:,2) == this_day;
        this_cluster_idx =  psth_unique_avg_animal_group_good_RT_cumulative_indices(:,3) == cluster_idx;
        if sum(this_day_idx & this_cluster_idx) == 0
            continue
        end
        for_plot_mean_max_ampl = psth_good_RT_cumulative_mean_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_sem_max_ampl = psth_good_RT_cumulative_sem_max_ampl(this_day_idx & this_cluster_idx);
        for_plot_seq_length = psth_unique_avg_animal_group_good_RT_cumulative_indices(this_day_idx & this_cluster_idx, 1);
        
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
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['Str response vs cumulative for Cluster ' num2str(cluster_idx)])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% START trials back

num_trials_back = 10;
good_RT_range = [0.01, 0.5];
% test1 = [0 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1];
% for i=1:length(test1)
%     if i<=num_trials_back
%         test2(i) = sum(test1(1:i-1));
%     else
%         test2(i) = sum(test1(i-num_trials_back:i-1));
%     end
% end

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
    repmat(good_RT_trials_back_lengths_cell{rec_idx}', 1, n_depths(rec_idx))', ...
    1:num_recordings, 'UniformOutput', false);
for_psth_good_RT_trials_back_lengths = vertcat(for_psth_good_RT_trials_back_lengths_cell{:});

% - make indices for trials_back good RT sequence
% cluster id has nans!!!!!!!!!!!!
psth_good_RT_trials_back_group_indices = [for_psth_animal_ids(psth_use_days), for_psth_good_RT_trials_back_lengths(psth_use_days), ...
    for_psth_days_from_learning(psth_use_days), for_psth_cluster_ids(psth_use_days)];
[psth_unique_good_RT_trials_back_indices, ~, psth_good_RT_trials_back_indices] = unique(psth_good_RT_trials_back_group_indices, 'rows');

psth_grouped_cluster_good_RT_trials_back = ap.groupfun(@sum, ...
    cat_flattened_stim_binned_spikes(psth_use_days, :), psth_good_RT_trials_back_indices, []);

% - norm and smooth sort RT split trial
psth_grouped_cluster_good_RT_trials_back_smooth_norm = cell2mat(arrayfun(@(rep) ...
    normalize_and_smooth(psth_grouped_cluster_good_RT_trials_back(rep, :)), ...
    1:size(psth_grouped_cluster_good_RT_trials_back, 1), ...
    'UniformOutput', false)');

% - group by ld and RT sort split trial
psth_good_RT_trials_back_day_trial_indices = [psth_unique_good_RT_trials_back_indices(:, 2), ...
    psth_unique_good_RT_trials_back_indices(:, 3), ...
    psth_unique_good_RT_trials_back_indices(:, 4)];
[psth_unique_avg_animal_group_good_RT_trials_back_indices, ~, ...
    psth_animal_group_clusters_good_RT_trials_back_indices] = ...
    unique(psth_good_RT_trials_back_day_trial_indices, 'rows');

avg_psth_grouped_good_RT_trials_back_smooth_norm = ap.groupfun(@mean, ...
    psth_grouped_cluster_good_RT_trials_back_smooth_norm, ....
    psth_animal_group_clusters_good_RT_trials_back_indices, []);

% count animals
num_animals_stim_psth_good_RT_trials_back = accumarray(psth_animal_group_clusters_good_RT_trials_back_indices, 1);

% - get max ampl for good trials_back
psth_window_for_max = psth_stim_time>0 & psth_stim_time<0.3;
psth_all_grouped_good_RT_trials_back_max_ampl = max(psth_grouped_cluster_good_RT_trials_back_smooth_norm(:, psth_window_for_max), [], 2);

% group by ld and seq length
psth_good_RT_trials_back_mean_max_ampl = ap.groupfun(@nanmean, ...
    psth_all_grouped_good_RT_trials_back_max_ampl, psth_animal_group_clusters_good_RT_trials_back_indices);

% do sem for errorbar
psth_good_RT_trials_back_std_max_ampl = ap.groupfun(@nanstd, ...
    psth_all_grouped_good_RT_trials_back_max_ampl, psth_animal_group_clusters_good_RT_trials_back_indices);
psth_good_RT_trials_back_sem_max_ampl = psth_good_RT_trials_back_std_max_ampl ./ sqrt(num_animals_stim_psth_good_RT_trials_back);

% - plot trials_back
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = psth_unique_days_from_learning(day_idx);
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
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['Str response vs trials back for Cluster ' num2str(cluster_idx)])
end
%% END trials back

%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    this_day_idx = for_wf_days_from_learning(wf_use_days) == this_day;
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
    for_wf_RT_cluster_day = for_wf_RT(for_wf_days_from_learning(wf_use_days) == this_day);
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

%% get ROIs
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
    sgtitle(['WF ROI for Cluster ' num2str(cluster_idx)])
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
unique_plot_day_values = unique(plot_day_values);
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
    title('WF ROI max amplitude mean')
    sgtitle(['Cluster ' num2str(cluster_idx)]);
end


%%%%%%%%%%%%%%%
%% split into 3 by RT sorting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get sort idx for RT

num_RT_split_trials = 5;
for rec_idx=1:num_recordings
    % (to discretize by RT rank)
    rank_RT_per_rec = tiedrank(all_RT{rec_idx});
    for_wf_sort_RT_split_trial_ids_cell{rec_idx} = discretize(rank_RT_per_rec,...
        round(linspace(1,max(rank_RT_per_rec),num_RT_split_trials+1)))';

    % (to discretize by RT bin)
%     discretize_RT_bin = discretize(all_RT{rec_idx},[-Inf,0:0.2:0.6,Inf]);
end
for_wf_sort_RT_split_trial_ids = horzcat(for_wf_sort_RT_split_trial_ids_cell{:})';

wf_per_rec_sorted_RT = vertcat(all_RT{:});

%% make sort RT split trial indices
%%%%%% LEFT HERE %%%%%%%%%%%%%%%%%%
wf_sort_RT_split_trial_group_indices = [for_wf_animal_ids(wf_use_days), ...
    for_wf_days_from_learning(wf_use_days) , for_wf_sort_RT_split_trial_ids(wf_use_days)];
[wf_unique_sort_RT_split_trial_indices, ~, wf_sort_RT_split_trial_indices] = unique(wf_sort_RT_split_trial_group_indices, 'rows');

wf_RT_split_trial_RT_mean = ap.groupfun(@mean, ...
    wf_per_rec_sorted_RT(wf_use_days), wf_sort_RT_split_trial_indices);

grouped_sort_RT_split_trial_norm_stim_Vs = ap.groupfun(@mean, ...
    reshape_temp_all_norm_V_stim_align, [], [], wf_sort_RT_split_trial_indices);

%% group by ld and split trial
wf_sort_RT_day_trial_indices = [wf_unique_sort_RT_split_trial_indices(:, 2), wf_unique_sort_RT_split_trial_indices(:, 3)];
[wf_unique_avg_animal_sort_RT_group_indices, ~, wf_animal_sort_RT_group_clusters_indices] = unique(wf_sort_RT_day_trial_indices, 'rows');

avg_grouped_sort_RT_split_trial_norm_stim_Vs = ap.groupfun(@mean, ...
    grouped_sort_RT_split_trial_norm_stim_Vs, [], [], wf_animal_sort_RT_group_clusters_indices);

avg_wf_RT_split_trial_RT_mean = ap.groupfun(@mean, ...
    wf_RT_split_trial_RT_mean, wf_animal_sort_RT_group_clusters_indices);

%% count animals
num_animals_stim_wf = accumarray(wf_animal_sort_RT_group_clusters_indices, 1);

%% get ROIs
all_avg_manual_kernel_roi_sort_RT = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    this_ROI = manual_kernel_ROIs{cluster_idx};
    all_avg_manual_kernel_roi_sort_RT{cluster_idx} = squeeze(ap.wf_roi(U_master,avg_grouped_sort_RT_split_trial_norm_stim_Vs, [], [], this_ROI));
end

%% plot RT sort split trial ROIs
my_colormap = ap.colormap('KG', num_RT_split_trials);
for cluster_idx=1:num_clusters
    figure;
    RT_ROIs = tiledlayout('flow');
    for day_idx=1:length(wf_unique_days_from_learning)
        this_day = wf_unique_days_from_learning(day_idx);
        this_day_idx = wf_unique_avg_animal_sort_RT_group_indices(:,1) == this_day;
        for_plot_manual_kernel_roi_sort_RT = all_avg_manual_kernel_roi_sort_RT{cluster_idx}(:, this_day_idx);
        for_plot_mean_RT = avg_wf_RT_split_trial_RT_mean(this_day_idx);

        nexttile;
        plot(wf_stim_time, for_plot_manual_kernel_roi_sort_RT);
        hold on;
        xline(for_plot_mean_RT)
        title(['Day ' num2str(wf_unique_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['RT sort split WF ROI for Cluster ' num2str(cluster_idx)])
    linkaxes(RT_ROIs.Children)
    xlim([-0.5 1])
end

% %% plot both split trial and sort RT split trial
% 
% my_colormap = ap.colormap('KR', num_split_trials);
% for cluster_idx=1:num_clusters
%     figure;
%     tiledlayout(length(psth_unique_days_from_learning), 2);
%     for day_idx=1:length(psth_unique_days_from_learning)
%         this_day = psth_unique_days_from_learning(day_idx); 
%         this_day_idx = psth_unique_avg_animal_group_sort_RT_indices(:,2) == this_day;
%         this_cluster_idx =  psth_unique_avg_animal_group_sort_RT_indices(:,3) == cluster_idx;
%         if sum(this_day_idx & this_cluster_idx) == 0
%             continue
%         end
%         
%         for_plot_RT_psth = avg_psth_grouped_sort_RT_split_trial_smooth_norm(this_day_idx & this_cluster_idx, :);
%         for_plot_mean_RT = avg_psth_RT_split_trial_RT_mean(this_day_idx & this_cluster_idx);
%         nexttile;
%         plot(psth_stim_time, for_plot_RT_psth);
%         ylim([-0.5 4]);
%         hold on;
%         xline(for_plot_mean_RT)
%         title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
%         colororder(gca, my_colormap);
% 
%         for_plot_psth = avg_psth_grouped_split_trial_smooth_norm(this_day_idx & this_cluster_idx, :);
%         nexttile;
%         plot(psth_stim_time, for_plot_psth);
%         ylim([-0.5 4]);
%         title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
%         colororder(gca, my_colormap);
%     end
%     sgtitle(['RT split and trial split Cluster ' num2str(cluster_idx)])
% end

%% max ampl
all_grouped_manual_kernel_roi_sort_RT = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    this_ROI = manual_kernel_ROIs{cluster_idx};
    all_grouped_manual_kernel_roi_sort_RT{cluster_idx} = squeeze(ap.wf_roi(U_master,grouped_sort_RT_split_trial_norm_stim_Vs, [], [], this_ROI));
end

wf_window_for_max = wf_stim_time>0 & wf_stim_time<0.3;
wf_sort_RT_all_grouped_max_ampl = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    wf_sort_RT_all_grouped_max_ampl{cluster_idx} = max(all_grouped_manual_kernel_roi_sort_RT{cluster_idx}(wf_window_for_max, :), [], 1);
end

% group by ld and split trial
wf_sort_RT_mean_max_ampl = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    wf_sort_RT_mean_max_ampl{cluster_idx} = ap.groupfun(@mean, ...
        wf_sort_RT_all_grouped_max_ampl{cluster_idx}, [], wf_animal_sort_RT_group_clusters_indices);
end

% do sem for errorbar
wf_sort_RT_std_max_ampl = cell(num_clusters, 1);
wf_sort_RT_sem_max_ampl = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    wf_sort_RT_std_max_ampl{cluster_idx} = ap.groupfun(@nanstd, ...
        wf_sort_RT_all_grouped_max_ampl{cluster_idx}, [], wf_animal_sort_RT_group_clusters_indices);
    wf_sort_RT_sem_max_ampl{cluster_idx} = wf_sort_RT_std_max_ampl{cluster_idx}' ./ sqrt(num_animals_stim_wf);
end

%% plot max ampl

curr_color = 'k';

plot_day_values = wf_unique_avg_animal_sort_RT_group_indices(:,1); % Extract x values
unique_plot_day_values = unique(plot_day_values);
plot_trial_values = wf_unique_avg_animal_sort_RT_group_indices(:,2); % Extract y values

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
        y_positions{cluster_idx} = [y_positions{cluster_idx}, wf_sort_RT_mean_max_ampl{cluster_idx}(indices)];
        x_labels_wf_max{cluster_idx} = [x_labels_wf_max{cluster_idx}; arrayfun(@(a, b) sprintf('(%d,%d)', a, b), ...
            plot_day_values(indices), plot_trial_values(indices), 'UniformOutput', false)];
        error_vals{cluster_idx} = [error_vals{cluster_idx}, wf_sort_RT_sem_max_ampl{cluster_idx}(indices)'];

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
    title('WF ROI max amplitude mean')
    sgtitle(['Cluster ' num2str(cluster_idx)]);
end


%% sort RT vs wf response per day 
for cluster_idx=1:num_clusters
    figure;
    tiledlayout('flow');
    for day_idx=1:length(psth_unique_days_from_learning)
        this_day = wf_unique_days_from_learning(day_idx);
        this_day_idx = wf_unique_avg_animal_sort_RT_group_indices(:,1) == this_day;
        for_plot_mean_max_ampl = wf_sort_RT_mean_max_ampl{cluster_idx}(this_day_idx);
        for_plot_sem_max_ampl = wf_sort_RT_sem_max_ampl{cluster_idx}(this_day_idx);
        for_plot_mean_RT = avg_wf_RT_split_trial_RT_mean(this_day_idx);

        nexttile;
%         plot(for_plot_mean_RT, for_plot_mean_max_ampl);
         errorbar(for_plot_mean_RT, for_plot_mean_max_ampl, for_plot_sem_max_ampl, 'o', 'CapSize', 0, ...
        'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
%         ylim([-0.5 4]);
%         hold on;
%         xline(for_plot_mean_RT)
        yline(0)
        ylabel('Max ampl of str response')
        xlabel('Mean RT in group')
        title(['Day ' num2str(psth_unique_days_from_learning(day_idx))])
        colororder(gca, my_colormap);
    end
    sgtitle(['RT vs WF response for ROI ' num2str(cluster_idx)])
end
 
% 
% my_colormap = ap.colormap('KG', num_RT_split_trials);
% for cluster_idx=1:num_clusters
%     figure;
%     tiledlayout('flow');
%     for day_idx=1:length(wf_unique_days_from_learning)
%         this_day = wf_unique_days_from_learning(day_idx);
%         this_day_idx = wf_unique_avg_animal_sort_RT_group_indices(:,1) == this_day;
%         for_plot_manual_kernel_roi_sort_RT = all_avg_manual_kernel_roi_sort_RT{cluster_idx}(:, this_day_idx);
%         for_plot_mean_RT = avg_wf_RT_split_trial_RT_mean(this_day_idx);
% 
%         nexttile;
%         plot(wf_stim_time, for_plot_manual_kernel_roi_sort_RT);
%         hold on;
%         xline(for_plot_mean_RT)
%         title(['Day ' num2str(wf_unique_days_from_learning(day_idx))])
%         colororder(gca, my_colormap);
%     end
%     sgtitle(['RT sort split WF ROI for Cluster ' num2str(cluster_idx)])
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
