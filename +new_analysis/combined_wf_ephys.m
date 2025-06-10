%% load
save_path = '\\qnap-ap001.dpag.ox.ac.uk\APlab\Users\Andrada-Maria_Marica\long_str_ctx_data';

bhv_data_path = fullfile(save_path, "swr_bhv.mat");
wf_data_path = fullfile(save_path, "ctx_wf.mat");
ephys_data_path = fullfile(save_path, "ephys.mat");
ctx_str_maps_data_path = fullfile(save_path, 'ctx_maps_to_str.mat');

load(bhv_data_path)
load(wf_data_path)
load(ephys_data_path)
load(ctx_str_maps_data_path)

master_U_fn = fullfile(save_path,'U_master.mat');
load(master_U_fn, 'U_master');

load(fullfile(save_path, "all_ROIs.mat"));
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

%% use kernel ROIs on wf data
%% - random
num_wf_recordings = height(wf);
unique_stims_nan = unique(vertcat(wf.trial_stim_values{:}));
unique_stims = unique_stims_nan(~isnan(unique_stims_nan));
wf_stim_time = wf.wf_stim_time{1};

%% - Get avg Vs and baseline subtract

% group Vs per stim
all_avg_stim_Vs = cell(numel(unique_stims), 1);
for stim_idx = 1:numel(unique_stims)
    stim_grouped_Vs = arrayfun(@(rec_idx) ...
        wf.V_no_move_stim_align{rec_idx}(wf.trial_stim_values{rec_idx} == unique_stims(stim_idx), :, :), ...
        1:num_wf_recordings, 'UniformOutput', false);
    temp_avg_stim_grouped_Vs = cellfun(@(rec_V) ...
        squeeze(mean(rec_V, 1)), ...
        stim_grouped_Vs, 'UniformOutput', false);
    all_avg_stim_Vs {stim_idx} = cat(3, temp_avg_stim_grouped_Vs{:});
end

% get baseline
baseline_idx = wf_stim_time > -0.2 & wf_stim_time < 0;
all_stim_baseline = cellfun(@(V) mean(V(baseline_idx, :, :), 1), all_avg_stim_Vs, 'UniformOutput', false);
avg_stim_baseline = squeeze(mean(cell2mat(all_stim_baseline), 1));

all_norm_avg_stim_Vs = cellfun(@(V) V - permute(repmat(avg_stim_baseline, [1, 1, length(wf_stim_time)]), [3, 1, 2]), all_avg_stim_Vs, 'UniformOutput', false);


%% - group Vs
% get grouping by learning day
[~, ~, for_wf_animal_ids] = unique(bhv.animal);
for_wf_days_from_learning = bhv.days_from_learning;
for_wf_combined_days_from_learning = for_wf_days_from_learning;
for_wf_combined_days_from_learning(for_wf_combined_days_from_learning<-3) = -3;
for_wf_combined_days_from_learning(for_wf_combined_days_from_learning>2) = 2;

use_days = ~isnan(for_wf_combined_days_from_learning);
[wf_unique_avg_animal_group_indices, ~, wf_animal_group_clusters_indices] = unique(for_wf_combined_days_from_learning(use_days));

% avg across animals
avg_grouped_norm_stim_Vs = cell(numel(unique_stims), 1);
for stim_idx = 1:numel(unique_stims)
    avg_grouped_norm_stim_Vs{stim_idx} = ap.groupfun(@nanmean, ...
        all_norm_avg_stim_Vs{stim_idx}(:,:,use_days), [], [], wf_animal_group_clusters_indices);
end

for_plot_Vs = cellfun(@(x) permute(x, [2, 1, 3]), avg_grouped_norm_stim_Vs, 'UniformOutput', false);

%% - count animals
num_animals_stim_wf = cell(numel(unique_stims), 1);
for stim_idx = 1:numel(unique_stims)
    num_animals_stim_wf{stim_idx} = accumarray(wf_animal_group_clusters_indices, 1);
end

%% - get manual cluster ROIs

all_norm_stim_manual_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    this_ROI = manual_kernel_ROIs{cluster_idx};
    all_norm_stim_manual_kernel_roi{cluster_idx} = cellfun(@(V) cell2mat(arrayfun(@(rec_idx) ...
        ap.wf_roi(U_master,V(:,:, rec_idx)',[],[],this_ROI), ...
        1:num_wf_recordings, 'UniformOutput', false)'), ...
        all_norm_avg_stim_Vs, 'UniformOutput', false);
end

avg_grouped_norm_stim_manual_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        avg_grouped_norm_stim_manual_kernel_roi{cluster_idx}{stim_idx} = ap.groupfun(@nanmean, ...
            all_norm_stim_manual_kernel_roi{cluster_idx}{stim_idx}(use_days, :)', [], wf_animal_group_clusters_indices);
    end
end

% do sem for errorbar
std_grouped_norm_stim_manual_kernel_roi = cell(num_clusters, 1);
sem_grouped_norm_stim_manual_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        std_grouped_norm_stim_manual_kernel_roi{cluster_idx}{stim_idx} = ap.groupfun(@nanstd, ...
            all_norm_stim_manual_kernel_roi{cluster_idx}{stim_idx}(use_days, :)', [], wf_animal_group_clusters_indices);
        sem_grouped_norm_stim_manual_kernel_roi{cluster_idx}{stim_idx} = ...
            std_grouped_norm_stim_manual_kernel_roi{cluster_idx}{stim_idx} ./ sqrt(num_animals_stim_wf{stim_idx})';
    end
end

%% - PLOT ROI for manual cluster ROIs
days_for_plot = -3:2;
all_colormap = ap.colormap('BKR', 2*max(abs(days_for_plot))+1);
colormap_days = -max(abs(days_for_plot)):max(abs(days_for_plot));
these_days_from_learning = wf_unique_avg_animal_group_indices;
plot_day_idx = ismember(these_days_from_learning, days_for_plot);
% get right colours
plotted_days = these_days_from_learning(plot_day_idx);
my_colormap = all_colormap(ismember(colormap_days, plotted_days), :);

for cluster_idx=1:num_clusters

    figure;
    tiledlayout('flow')

    nexttile;
    imagesc(manual_kernel_ROIs{cluster_idx})
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    title('ROI used');

    for stim_idx=1:numel(unique_stims)

        for_plot_wf_roi = avg_grouped_norm_stim_manual_kernel_roi{cluster_idx}{stim_idx}(:, plot_day_idx);

        % get num animals for legend
        num_animals_plotted = num_animals_stim_wf{stim_idx}(plot_day_idx);

        % make legend
        legend_for_plot = arrayfun(@(day, num) ['Day ' num2str(day) ' (n = ' num2str(num) ')'], ...
            plotted_days, num_animals_plotted, 'UniformOutput', false);

        nexttile;
        plot(wf_stim_time, for_plot_wf_roi);
        colororder(gca, my_colormap);
        legend(legend_for_plot);
        xline(0, 'LineWidth', 2);

        %     ylim([-0.5 2.5])

        title(['Manual cluster ROI ' num2str(cluster_idx) ' for stim ' num2str(unique_stims(stim_idx))])
    end
    sgtitle(['Manual cluster ' num2str(cluster_idx)])
end

%% - max amplitude for manual kernels
max_ampl_window = wf_stim_time > 0 & wf_stim_time < 0.3;

% get max per rec
all_max_ampl_manual_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        all_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx} = ...
            max(all_norm_stim_manual_kernel_roi{cluster_idx}{stim_idx}(:, max_ampl_window)', [], 1);
    end
end

% group and get mean across mice
% !!!!!!! this is doing something funny, the output give values 1-12 but
% should be x 10-4
mean_max_ampl_manual_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        mean_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx} = ap.groupfun(@nanmean, ...
            all_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx}(use_days)', wf_animal_group_clusters_indices);
    end
end

% get median
median_max_ampl_manual_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        median_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx} = ap.groupfun(@median, ...
            all_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx}(use_days)', wf_animal_group_clusters_indices);
    end
end

% do sem for errorbar
std_max_ampl_manual_kernel_roi = cell(num_clusters, 1);
sem_max_ampl_manual_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        std_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx} = ap.groupfun(@nanstd, ...
            all_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx}(use_days)', wf_animal_group_clusters_indices);
        sem_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx} = std_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx} ./ sqrt(num_animals_stim_wf{stim_idx});
    end
end

%% plot max amplitude for manual kernel ROIs (mean)
days_for_plot = -3:2;
wf_curr_color = [0 0.7 0];
these_days_from_learning = wf_unique_avg_animal_group_indices;
plot_day_idx = ismember(these_days_from_learning, days_for_plot);

for cluster_idx=1:num_clusters

    figure;
    tiledlayout('flow')

    nexttile;
    imagesc(manual_kernel_ROIs{cluster_idx})
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    title('ROI used');
    for stim_idx=1:length(unique_stims)

        for_plot_mean_max_manual_kernel_roi = mean_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx};
        for_plot_sem_max_manual_kernel_roi = sem_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx};

        plotted_days = these_days_from_learning(plot_day_idx);

        %     % get num animals for legend
        %     num_animals_plotted = num_animals_stim_wf{stim_idx}(plot_day_idx);
        %
        %     % make legend
        %     legend_for_plot = arrayfun(@(day, num) ['Day ' num2str(day) ' (n = ' num2str(num) ')'], ...
        %         plotted_days, num_animals_plotted, 'UniformOutput', false);

        nexttile
        errorbar(plotted_days, for_plot_mean_max_manual_kernel_roi(plot_day_idx), for_plot_sem_max_manual_kernel_roi(plot_day_idx), '-o', 'CapSize', 0, ...
            'MarkerFaceColor', wf_curr_color , 'MarkerEdgeColor', wf_curr_color , 'Color', wf_curr_color );
        title(['Stim ' num2str(unique_stims(stim_idx))])
%         ylim([3*10^(-3), 12*10^(-3)])
    end
    sgtitle(['Max amplitude for ROI ' num2str(cluster_idx)])
end


%% PSTHS
num_psth_recordings = height(ephys);

%% - make big vectors of days from learning and mouse id
n_depths = arrayfun(@(rec_idx) ...
    size(all_ctx_maps_to_str.cortex_kernel_px{rec_idx}, 3) * ~isempty(all_ctx_maps_to_str.cortex_kernel_px{rec_idx}), ...
    1:height(all_ctx_maps_to_str));

[~, ~, for_psth_animal_ids] = unique(repelem(bhv.animal, n_depths));
for_psth_days_from_learning = repelem(bhv.days_from_learning, n_depths);
for_psth_combined_days_from_learning = for_psth_days_from_learning;
for_psth_combined_days_from_learning(for_psth_days_from_learning<-3) = -3;
for_psth_combined_days_from_learning(for_psth_days_from_learning>2) = 2;

%% - get 'fast' learning mice and 'slow' learning mice
unique_animals = unique(bhv.animal);
per_mouse_days_from_learning = cell(length(unique_animals) , 1);
min_mouse_ld = nan(length(unique_animals) , 1);
for animal_idx=1:length(unique_animals) 
    per_mouse_days_from_learning{animal_idx} = bhv.days_from_learning(strcmp(bhv.animal, unique_animals(animal_idx)));
    min_mouse_ld(animal_idx) = min(per_mouse_days_from_learning{animal_idx});
end
figure;
histogram(min_mouse_ld)

%% - get num trials
num_trials = cell(numel(unique_stims), 1);
for stim_idx = 1:numel(unique_stims)
    num_trials{stim_idx} = arrayfun(@(rec_idx) ...
        sum(ephys.trial_stim_values{rec_idx} == unique_stims(stim_idx)), ...
        1:num_psth_recordings);
end
for_psth_num_trials = cellfun(@(x) repelem(x, n_depths), num_trials, 'UniformOutput', false);

%% - get psths

all_avg_stim_grouped_binned_spikes = cell(numel(unique_stims), 1);
for stim_idx = 1:numel(unique_stims)
    stim_grouped_binned_spikes = arrayfun(@(rec_idx) ...
        ephys.binned_msn_spikes_stim_align{rec_idx}(ephys.trial_stim_values{rec_idx} == unique_stims(stim_idx), :, :), ...
        1:num_psth_recordings, 'UniformOutput', false);
    avg_stim_grouped_binned_spikes = cellfun(@(rec_spikes) ...
        squeeze(mean(rec_spikes, 1)), ...
        stim_grouped_binned_spikes, 'UniformOutput', false);
    all_avg_stim_grouped_binned_spikes {stim_idx} = horzcat(avg_stim_grouped_binned_spikes{:});
end

%% - group psths according to animal, cluster and day from learning
% only use nonnan learning days
use_days = ~isnan(for_psth_combined_days_from_learning);

[unique_cluster_ids, ~, ~] = unique(cluster_ids(use_days));
[unique_days_from_learning, ~, ~] = unique(for_psth_combined_days_from_learning(use_days));
[unique_animal_ids, ~, ~] = unique(for_psth_animal_ids(use_days));

% Combine cluster_ids and days into a single matrix for indexing
group_indices = [for_psth_animal_ids(use_days), cluster_ids(use_days), for_psth_combined_days_from_learning(use_days)];
[group_indices_unique_clusters, ~, group_clusters_indices] = unique(group_indices, 'rows');

all_sum_avg_stim_grouped_spikes = cell(numel(unique_stims), 1);
for stim_idx = 1:numel(unique_stims)
    all_sum_avg_stim_grouped_spikes{stim_idx} = ap.groupfun(@sum, ...
        all_avg_stim_grouped_binned_spikes{stim_idx}(:, use_days), [], group_clusters_indices);
end

% group num trials - when added combined days changed from unique to mean
grouped_num_trials = cell(numel(unique_stims), 1);
for stim_idx=1:numel(unique_stims)
    grouped_num_trials{stim_idx} = accumarray(group_clusters_indices, for_psth_num_trials{stim_idx}(use_days)', ...
        [length(unique(group_clusters_indices)), 1], @mean);
end

%% - get baseline
psth_stim_time = -0.5:0.001:1;
baseline_idx = psth_stim_time >= -0.5 & psth_stim_time <= -0.2;

per_stim_baseline_avg = cellfun(@(x) mean(x(baseline_idx, :), 1), all_sum_avg_stim_grouped_spikes, 'UniformOutput', false);
total_num_trials = cell2mat(grouped_num_trials')';
weighted_sum_stim_baseline = cell2mat(cellfun(@(per_stim_baseline, num_trials) per_stim_baseline .* num_trials', ...
    per_stim_baseline_avg, grouped_num_trials, 'UniformOutput', false));
weighted_avg_stim_baseline = sum(weighted_sum_stim_baseline, 1) ./ sum(total_num_trials, 1);

%% - norm smooth psths
gauss_win = gausswin(51, 3)';
softnorm = 10;
normalize_and_smooth = @(psth, rep) ...
    filter(gauss_win, sum(gauss_win), (psth - weighted_avg_stim_baseline(rep)) ./ (weighted_avg_stim_baseline(rep)+softnorm), [], 1);

all_smooth_stim_psths = cell(numel(unique_stims), 1);
for stim_idx=1:numel(unique_stims)
    all_smooth_stim_psths{stim_idx} = cell2mat(arrayfun(@(rep) ...
        normalize_and_smooth(squeeze(all_sum_avg_stim_grouped_spikes{stim_idx}(:, rep)), rep), ...
        1:size(all_sum_avg_stim_grouped_spikes{stim_idx}, 2), ...
        'UniformOutput', false));
end

%% - plot psth per mouse for each stim
stim_idx = 3;
cluster_idx = 1;
% for cluster_idx =1:num_clusters
for animal_idx=1:length(unique_animals)
    figure;
    per_mouse_psth_fig = tiledlayout('flow');
    for day_idx=1:length(unique_days_from_learning)
        this_day = unique_days_from_learning(day_idx);
        this_mouse_idx = group_indices_unique_clusters(:,1) == animal_idx & ...
            group_indices_unique_clusters(:,2) == cluster_idx & ...
            group_indices_unique_clusters(:,3) == this_day;
        nexttile;

        for stim_idx=1:numel(unique_stims)

            %         if min_mouse_ld(animal_idx) < -3

            if isempty(find(this_mouse_idx))
                continue
            end
            this_smooth_stim_psth = all_smooth_stim_psths{stim_idx}(:,  this_mouse_idx);
            plot(psth_stim_time, this_smooth_stim_psth)
            hold on;
            %             end
        end
        legend(arrayfun(@(x) sprintf('Stim %d', x), unique_stims, 'UniformOutput', false))
        title(['Day ' num2str(this_day)])
    end
    linkaxes(per_mouse_psth_fig.Children)
    sgtitle(sprintf('Cluster %d Animal %d\npre LD %d', ...
        cluster_idx, animal_idx, min_mouse_ld(animal_idx)));
end
% end


%% - average psths across mice

% mouse idx is first
avg_animal_group_indices = group_indices_unique_clusters(:,2:3);
[unique_avg_animal_group_indices, ~, animal_group_clusters_indices] = unique(avg_animal_group_indices, 'rows');

avg_smooth_stim_psths = cell(numel(unique_stims), 1);
for stim_idx=1:numel(unique_stims)
    avg_smooth_stim_psths{stim_idx} = ap.groupfun(@mean, ...
        all_smooth_stim_psths{stim_idx}, [], animal_group_clusters_indices);
end

num_animals_stim_psths = cell(numel(unique_stims), 1);
for stim_idx = 1:numel(unique_stims)
    num_animals_stim_psths{stim_idx} = accumarray(animal_group_clusters_indices, 1);
end

% do sem for errorbar
std_smooth_stim_psths = cell(numel(unique_stims), 1);
sem_smooth_stim_psths = cell(numel(unique_stims), 1);
for stim_idx=1:numel(unique_stims)
    std_smooth_stim_psths{stim_idx} = ap.groupfun(@std, ...
        all_smooth_stim_psths{stim_idx}, [], animal_group_clusters_indices);
    sem_smooth_stim_psths{stim_idx} = std_smooth_stim_psths{stim_idx} ./ sqrt(num_animals_stim_psths{stim_idx})';
end


%% - psths plot all days

days_for_plot = -3:2;
all_colormap = ap.colormap('BKR', 2*max(abs(days_for_plot))+1);
colormap_days = -max(abs(days_for_plot)):max(abs(days_for_plot));

for cluster_idx =1:num_clusters

    figure
    tiledlayout('flow')

    for stim_idx=1:length(unique_stims)

        this_cluster_psth_idx = unique_avg_animal_group_indices(:,1) == cluster_idx;
        these_days_from_learning = unique_avg_animal_group_indices(this_cluster_psth_idx, 2);

        plot_day_idx = ismember(these_days_from_learning, days_for_plot);
        for_plot_psth = avg_smooth_stim_psths{stim_idx}(:, this_cluster_psth_idx);

        % get right colours
        plotted_days = these_days_from_learning(plot_day_idx);
        my_colormap = all_colormap(ismember(colormap_days, plotted_days), :);

        % time
        x = psth_stim_time;

        % get num animals for legen
        num_animals_all = num_animals_stim_psths{stim_idx}(this_cluster_psth_idx);
        num_animals_plotted = num_animals_all(plot_day_idx);

        % make legend
        legend_for_plot = arrayfun(@(day, num) ['Day ' num2str(day) ' (n = ' num2str(num) ')'], ...
            plotted_days, num_animals_plotted, 'UniformOutput', false);

        nexttile;
        plot(x, for_plot_psth(:, plot_day_idx));
        colororder(gca, my_colormap);
        legend(legend_for_plot);
        xline(0, 'LineWidth', 2);
        ylim([-0.5 2.5])
        title(['MSNs Stim ' num2str(unique_stims(stim_idx))])

        %
        %     x_fill = [x, fliplr(x)];  % X values: time points and reversed time points
        %     y_fill = [avg + sem, fliplr(avg - sem)];  % Y values: upper and lower bounds of SEM
        %
        %     % Plot the shaded SEM region
        %     fill(x_fill, y_fill, 'k', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    sgtitle(['Cluster' num2str(cluster_idx)])

end

%% - get max amplitude across days

max_window = psth_stim_time>0 & psth_stim_time<0.3;

max_smooth_stim_psths = arrayfun(@(stim_idx) ...
    max(all_smooth_stim_psths{stim_idx}(max_window, :), [], 1), ...
    1:numel(unique_stims), 'UniformOutput', false);

% use avg_animal_group_indices to find across animal avg
mean_max_smooth_stim_psths = cell(numel(unique_stims), 1);
for stim_idx=1:numel(unique_stims)
    mean_max_smooth_stim_psths{stim_idx} = ap.groupfun(@mean, ...
        max_smooth_stim_psths{stim_idx}, [], animal_group_clusters_indices);
end

% median
median_max_smooth_stim_psths = cell(numel(unique_stims), 1);
for stim_idx=1:numel(unique_stims)
    median_max_smooth_stim_psths{stim_idx} = ap.groupfun(@median, ...
        max_smooth_stim_psths{stim_idx}, [], animal_group_clusters_indices);
end

% find sem for max
std_max_smooth_stim_psths = cell(numel(unique_stims), 1);
sem_max_smooth_stim_psths = cell(numel(unique_stims), 1);
for stim_idx=1:numel(unique_stims)
    std_max_smooth_stim_psths{stim_idx} = ap.groupfun(@std, ...
        max_smooth_stim_psths{stim_idx}, [], animal_group_clusters_indices);
    sem_max_smooth_stim_psths{stim_idx} = std_max_smooth_stim_psths{stim_idx} ./ sqrt(num_animals_stim_psths{stim_idx})';
end

%% plot max ampl (mean)

days_for_plot = -3:2;
psth_curr_color = [0.7 0 0];

for stim_idx=1:length(unique_stims)
    figure;
    tiledlayout(num_clusters,2);
    for cluster_idx = 1:num_clusters

        nexttile;
        imagesc(squeeze(centroid_images(cluster_idx,:,:)))
        axis image;
        clim(max(abs(clim)).*[-1,1]*0.7);
        ap.wf_draw('ccf','k');
        colormap(ap.colormap('PWG'));
        ylabel(['Cluster ', num2str(cluster_idx)], 'FontSize', 14, 'FontWeight','bold');


        this_cluster_psth_idx = unique_avg_animal_group_indices(:,1) == cluster_idx;
        these_days_from_learning = unique_avg_animal_group_indices(this_cluster_psth_idx, 2);

        plot_day_idx = ismember(these_days_from_learning, days_for_plot);
        for_plot_max_mean = mean_max_smooth_stim_psths{stim_idx}(this_cluster_psth_idx);
        for_plot_max_sem = sem_max_smooth_stim_psths{stim_idx}(this_cluster_psth_idx);

        % get right colours
        plotted_days = these_days_from_learning(plot_day_idx);
        %     my_colormap = all_colormap(ismember(colormap_days, plotted_days), :);

        % get num animals for legen
        num_animals_all = num_animals_stim_psths{stim_idx}(this_cluster_psth_idx);
        num_animals_plotted = num_animals_all(plot_day_idx);

        % make legend
        legend_for_plot = arrayfun(@(day, num) ['Day ' num2str(day) ' (n = ' num2str(num) ')'], ...
            plotted_days, num_animals_plotted, 'UniformOutput', false);

        nexttile;
        errorbar(plotted_days, for_plot_max_mean(:, plot_day_idx), for_plot_max_sem(:, plot_day_idx), '-o', 'CapSize', 0, ...
            'MarkerFaceColor', psth_curr_color, 'MarkerEdgeColor', psth_curr_color, 'Color', psth_curr_color);
        ylim([0, 6])
        title(['Cluster ' num2str(cluster_idx)])

    end
    sgtitle(['MSNs Max psth amplitude for stim ' num2str(unique_stims(stim_idx))])
end
% ADD line per mouse in grey


%% per mouse max ampl
% stim_idx = 3;
% cluster_idx = 1;
% % for cluster_idx =1:num_clusters
% %     for stim_idx=1:numel(unique_stims)
%         for animal_idx=1:length(unique_animals)
%             if min_mouse_ld(animal_idx) > -3
%                 figure;
%                 tiledlayout('flow');
%                 this_mouse_idx = group_indices_unique_clusters(:,1) == animal_idx & ...
%                         group_indices_unique_clusters(:,2) == cluster_idx;
%                 these_days_from_learning = group_indices_unique_clusters(this_mouse_idx,3);
%                 if isempty(find(this_mouse_idx))
%                     continue
%                 end
%                 this_max_ampl = max_smooth_stim_psths{stim_idx}(this_mouse_idx);
%                 nexttile;
%                 plot(these_days_from_learning, this_max_ampl, '-o')
%                 hold on;
%                 xline(0)
%                 sgtitle(['Cluster ' num2str(cluster_idx) ...
%                     ' Animal ' num2str(animal_idx)])
%                 title(['pre LD ' num2str(min_mouse_ld(animal_idx))])
%             end
%         end
% %     end
% % end

%% combined plot

days_for_plot = -3:2;

psth_curr_color = [0.7 0 0];
wf_curr_color = [0 0.7 0];
wf_these_days_from_learning = wf_unique_avg_animal_group_indices;
wf_plot_day_idx = ismember(wf_these_days_from_learning, days_for_plot);

for stim_idx=1:length(unique_stims)

    figure;
    tiledlayout(num_clusters, 5)
    for cluster_idx=1:num_clusters

        % ROI plot
        nexttile;
        imagesc(manual_kernel_ROIs{cluster_idx})
        axis image;
        axis off;
        clim(max(abs(clim)).*[-1,1]*0.7);
        ap.wf_draw('ccf','k');
        colormap(ap.colormap('PWG'));
        title('ROI used');


        % max ampl plot
        for_plot_mean_max_manual_kernel_roi = mean_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx};
        for_plot_sem_max_manual_kernel_roi = sem_max_ampl_manual_kernel_roi{cluster_idx}{stim_idx};

        wf_plotted_days = wf_these_days_from_learning(wf_plot_day_idx);

        %     % get num animals for legend
        %     num_animals_plotted = num_animals_stim_wf{stim_idx}(plot_day_idx);
        %
        %     % make legend
        %     legend_for_plot = arrayfun(@(day, num) ['Day ' num2str(day) ' (n = ' num2str(num) ')'], ...
        %         plotted_days, num_animals_plotted, 'UniformOutput', false);

        nexttile;
        errorbar(wf_plotted_days, for_plot_mean_max_manual_kernel_roi(wf_plot_day_idx), for_plot_sem_max_manual_kernel_roi(wf_plot_day_idx), '-o', 'CapSize', 0, ...
            'MarkerFaceColor', wf_curr_color, 'MarkerEdgeColor', wf_curr_color, 'Color', wf_curr_color);
        ylim([2*10^(-4), 4*10^(-3)])
        xlim([days_for_plot(1) days_for_plot(end)])
        ylabel('{\Delta}F/F');
        xlabel('Days from association day');
        title('WF max')

        nexttile;
        imagesc(squeeze(centroid_images(cluster_idx,:,:)))
        axis image;
        axis off;
        clim(max(abs(clim)).*[-1,1]*0.7);
        ap.wf_draw('ccf','k');
        colormap(ap.colormap('PWG'));
        title(['Cluster ', num2str(cluster_idx)], 'FontSize', 14, 'FontWeight','bold');

        this_cluster_psth_idx = unique_avg_animal_group_indices(:,1) == cluster_idx;
        psth_these_days_from_learning = unique_avg_animal_group_indices(this_cluster_psth_idx, 2);

        psth_plot_day_idx = ismember(psth_these_days_from_learning, days_for_plot);
        for_plot_max_mean_psth = mean_max_smooth_stim_psths{stim_idx}(this_cluster_psth_idx);
        for_plot_max_sem_psth = sem_max_smooth_stim_psths{stim_idx}(this_cluster_psth_idx);

        % get right colours
        psth_plotted_days = psth_these_days_from_learning(psth_plot_day_idx);
        %     my_colormap = all_colormap(ismember(colormap_days, plotted_days), :);
        %
        %         % get num animals for legen
        %         num_animals_all = num_animals_stim_psths{stim_idx}(this_cluster_psth_idx);
        %         num_animals_plotted = num_animals_all(psth_plot_day_idx);
        %
        %         % make legend
        %         legend_for_plot = arrayfun(@(day, num) ['Day ' num2str(day) ' (n = ' num2str(num) ')'], ...
        %             plotted_days, num_animals_plotted, 'UniformOutput', false);

        nexttile;
        errorbar(psth_plotted_days, for_plot_max_mean_psth(:, psth_plot_day_idx), for_plot_max_sem_psth(:, psth_plot_day_idx), '-o', 'CapSize', 0, ...
            'MarkerFaceColor', psth_curr_color, 'MarkerEdgeColor', psth_curr_color, 'Color', psth_curr_color);
        ylim([0 4])
        xlim([days_for_plot(1) days_for_plot(end)])

        ylabel('{\Delta}R/R');
        xlabel('Days from association day');
        title('PSTH max ')

        % combined
        nexttile;
        yyaxis left
        errorbar(wf_plotted_days, for_plot_mean_max_manual_kernel_roi(wf_plot_day_idx), for_plot_sem_max_manual_kernel_roi(wf_plot_day_idx), '-o', 'CapSize', 0, ...
            'MarkerFaceColor', wf_curr_color, 'MarkerEdgeColor', wf_curr_color, 'Color', wf_curr_color);
        ylim([2*10^(-4), 4*10^(-3)])
        ylabel('WF max', 'Color', wf_curr_color)
        xlim([days_for_plot(1) days_for_plot(end)])
        ax = gca;
        ax.YColor = wf_curr_color;

        yyaxis right
        errorbar(psth_plotted_days, for_plot_max_mean_psth(:, psth_plot_day_idx), for_plot_max_sem_psth(:, psth_plot_day_idx), '-o', 'CapSize', 0, ...
            'MarkerFaceColor', psth_curr_color, 'MarkerEdgeColor', psth_curr_color, 'Color', psth_curr_color);
        ylim([0 3])
        xlim([days_for_plot(1) days_for_plot(end)])
        ylabel('PSTH max')
        ax = gca;
        ax.YColor = psth_curr_color;
    end
    sgtitle(['Stim ' num2str(unique_stims(stim_idx))])
end

%% FRAC resp

p_thresh = 0.01;
all_unit_resp_p_value = vertcat(ephys.unit_resp_p_value);

% ger resp units
all_resp_units = cell(numel(unique_stims), 1);
cat_all_resp_units = cell(numel(unique_stims), 1);
for stim_idx=1:length(unique_stims)
    all_resp_units{stim_idx} = arrayfun(@(rep) [all_unit_resp_p_value{rep}{stim_idx}{:}] < p_thresh, ...
        1:length(all_unit_resp_p_value), 'uni', false, 'ErrorHandler', @(x, varargin) []);
    cat_all_resp_units{stim_idx} = horzcat(all_resp_units{stim_idx}{:})';
end

% get single units
cat_all_single_units = vertcat(ephys.single_unit_idx{:});

% get cell type 
cat_all_msn_units = vertcat(ephys.str_msn_idx{:});

% get unit depth group
all_unit_depth_group = vertcat(ephys.unit_depth_group);

% get unit cluster id based on unit depth group
depth_group_edges = vertcat(ephys.depth_group_edges);
depth_indexes = cellfun(@(x) 1:numel(x)-1, depth_group_edges, ...
    'UniformOutput', false,'ErrorHandler', @(x, varargin) []);
all_depth_indexes = horzcat(depth_indexes{:});

% get per rec cluster ids 
per_rec_cluster_ids = mat2cell(cluster_ids, n_depths);

% get units cluster ids
unit_cluster_ids = cell(length(per_rec_cluster_ids), 1);
for rec_idx=1:length(unit_cluster_ids)
    unit_depth = all_unit_depth_group{rec_idx};
    cluster = per_rec_cluster_ids{rec_idx};

    this_unit_cluster_ids = zeros(size(unit_depth));
    this_unit_cluster_ids(~isnan(unit_depth)) = cluster(unit_depth(~isnan(unit_depth)));
    unit_cluster_ids{rec_idx} = this_unit_cluster_ids;
end
all_unit_cluster_ids = vertcat(unit_cluster_ids{:});

% make days from learning and animal grouping idx
n_units = arrayfun(@(rec_idx) ...
    size(ephys.unit_depth_group{rec_idx}, 1) * ~isempty(ephys.unit_depth_group{rec_idx}), ...
    1:height(ephys));
[~, ~, for_units_animal_ids] = unique(repelem(bhv.animal, n_units));
for_units_days_from_learning = repelem(bhv.days_from_learning, n_units);
for_units_combined_days_from_learning = for_units_days_from_learning;
for_units_combined_days_from_learning(for_units_days_from_learning<-3) = -3;
for_units_combined_days_from_learning(for_units_days_from_learning>2) = 2;

% - group per cluster
use_days_units = ~isnan(for_units_combined_days_from_learning);
unit_group_indices = [for_units_animal_ids(use_days_units), ...
    all_unit_cluster_ids(use_days_units), ...
    for_units_combined_days_from_learning(use_days_units), ...
    cat_all_msn_units(use_days_units)];
[unit_group_indices_unique_clusters, ~, unit_group_clusters_indices] = unique(unit_group_indices, 'rows');

% get frac resp units
num_resp_units = cell(numel(unique_stims), 1);
num_all_units = cell(numel(unique_stims), 1);
for stim_idx=1:length(unique_stims)
    num_resp_units{stim_idx} = ap.groupfun(@sum, ...
        cat_all_resp_units{stim_idx}(use_days_units),  unit_group_clusters_indices, []);
    num_all_units{stim_idx} = ap.groupfun(@length, ...
        cat_all_resp_units{stim_idx}(use_days_units),  unit_group_clusters_indices, []);
end
frac_resp_units = arrayfun(@(stim_idx) num_resp_units{stim_idx}./num_all_units{stim_idx}, ...
    1:length(unique_stims), 'UniformOutput', false);

% - group per mouse and get avg
unit_avg_animal_unit_group_indices = unit_group_indices_unique_clusters(:,2:4);
[unit_unique_avg_animal_group_indices, ~, unit_animal_group_clusters_indices] = unique(unit_avg_animal_unit_group_indices, 'rows');

num_animals_stim_unit = cell(numel(unique_stims), 1);
for stim_idx = 1:numel(unique_stims)
    num_animals_stim_unit{stim_idx} = accumarray(unit_animal_group_clusters_indices, 1);
end

avg_frac_resp_units = cell(numel(unique_stims), 1);
std_frac_resp_units = cell(numel(unique_stims), 1);
sem_frac_resp_units = cell(numel(unique_stims), 1);
for stim_idx=1:length(unique_stims)
    avg_frac_resp_units{stim_idx} = ap.groupfun(@mean, ...
        frac_resp_units{stim_idx}, unit_animal_group_clusters_indices, []);
    std_frac_resp_units{stim_idx} = ap.groupfun(@std, ...
        frac_resp_units{stim_idx}, unit_animal_group_clusters_indices, []);
    sem_frac_resp_units{stim_idx} = std_frac_resp_units{stim_idx} ./ sqrt(num_animals_stim_unit{stim_idx});
end

%% - plot frac resp

curr_color = 'k';
% max_colormap = ap.colormap('BKR', 2*max(abs(days_for_plot))+1);
% colormap_days = -max(abs(days_for_plot)):max(abs(days_for_plot));

for stim_idx=1:length(unique_stims)

    figure;
    tiledlayout(num_clusters,2);
    for cluster_id = 1:num_clusters

        nexttile;
        imagesc(squeeze(centroid_images(cluster_id,:,:)))
        axis image;
        clim(max(abs(clim)).*[-1,1]*0.7);
        ap.wf_draw('ccf','k');
        colormap(ap.colormap('PWG'));
        ylabel(['Cluster ', num2str(cluster_id)], 'FontSize', 14, 'FontWeight','bold');


        this_cluster_unit_idx = unit_unique_avg_animal_group_indices(:,1) == cluster_id;
        good_units = unit_unique_avg_animal_group_indices(:,3) == 1;
        these_days_from_learning = unit_unique_avg_animal_group_indices(good_units & this_cluster_unit_idx, 2);
        for_plot_avg_frac = avg_frac_resp_units{stim_idx}(this_cluster_unit_idx & good_units);
        for_plot_sem_frac = sem_frac_resp_units{stim_idx}(this_cluster_unit_idx & good_units);

        % get right colours
        %     my_colormap = all_colormap(ismember(colormap_days, plotted_days), :);

        % get num animals for legen
        num_animals_plotted = num_animals_stim_unit{stim_idx}(this_cluster_unit_idx & good_units);

        % make legend
        legend_for_plot = arrayfun(@(day, num) ['Day ' num2str(day) ' (n = ' num2str(num) ')'], ...
            these_days_from_learning, num_animals_plotted, 'UniformOutput', false);

        nexttile;
        errorbar(these_days_from_learning, for_plot_avg_frac, for_plot_sem_frac, '-o', 'CapSize', 0, ...
            'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);

        ylim([0,0.25])
        title(['Cluster ' num2str(cluster_id)])

    end
    frac_resp_title = ['Frac resp units for stim ' num2str(unique_stims(stim_idx))];
    sgtitle(frac_resp_title, 'FontSize', 20);
 end
