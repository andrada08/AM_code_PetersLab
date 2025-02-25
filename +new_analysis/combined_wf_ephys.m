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
manual_kernel_ROIs = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    figure;
    imagesc(kernel_ROIs{cluster_idx})
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    roi_poly = drawpolygon;
    manual_kernel_ROIs{cluster_idx} = createMask(roi_poly);
    title(['ROI for Cluster ', num2str(cluster_idx)]);
end

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
for_wf_days_from_learning = bhv.days_from_learning;
[~, ~, for_wf_animal_ids] = unique(bhv.animal);

use_days = ~isnan(for_wf_days_from_learning);
[wf_unique_avg_animal_group_indices, ~, wf_animal_group_clusters_indices] = unique(for_wf_days_from_learning(use_days));

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

%% - get cluster ROIs

all_norm_stim_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    this_ROI = kernel_ROIs{cluster_idx};
    all_norm_stim_kernel_roi{cluster_idx} = cellfun(@(V) cell2mat(arrayfun(@(rec_idx) ...
        ap.wf_roi(U_master,V(:,:, rec_idx)',[],[],this_ROI), ...
        1:num_wf_recordings, 'UniformOutput', false)'), ...
        all_norm_avg_stim_Vs, 'UniformOutput', false);
end

avg_grouped_norm_stim_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        avg_grouped_norm_stim_kernel_roi{cluster_idx}{stim_idx} = ap.groupfun(@nanmean, ...
            all_norm_stim_kernel_roi{cluster_idx}{stim_idx}(use_days, :)', [], wf_animal_group_clusters_indices);
    end
end

% do sem for errorbar
std_grouped_norm_stim_kernel_roi = cell(num_clusters, 1);
sem_grouped_norm_stim_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        std_grouped_norm_stim_kernel_roi{cluster_idx}{stim_idx} = ap.groupfun(@nanstd, ...
            all_norm_stim_kernel_roi{cluster_idx}{stim_idx}(use_days, :)', [], wf_animal_group_clusters_indices);
        sem_grouped_norm_stim_kernel_roi{cluster_idx}{stim_idx} = ...
            std_grouped_norm_stim_kernel_roi{cluster_idx}{stim_idx} ./ sqrt(num_animals_stim_wf{stim_idx})';
    end
end


%% - PLOT
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
    imagesc(kernel_ROIs{cluster_idx})
    axis image;
    axis off;
    clim(max(abs(clim)).*[-1,1]*0.7);
    ap.wf_draw('ccf','k');
    colormap(ap.colormap('PWG'));
    title('ROI used');

    for stim_idx=1:numel(unique_stims)



        for_plot_wf_roi = avg_grouped_norm_stim_kernel_roi{cluster_idx}{stim_idx}(:, plot_day_idx);

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

        title(['Cluster ROI for stim ' num2str(unique_stims(stim_idx))])
    end
    sgtitle(['Cluster ' num2str(cluster_idx)])
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


%% - PLOT
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

        title(['Manual cluster ROI for stim ' num2str(unique_stims(stim_idx))])
    end
    sgtitle(['Manual cluster ' num2str(cluster_idx)])
end

%% - max amplitude for manual kernels
max_ampl_window = wf_stim_time > 0 & wf_stim_time < 0.3;

% get max per rec
all_max_ampl_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        all_max_ampl_kernel_roi{cluster_idx}{stim_idx} = ...
            max(all_norm_stim_kernel_roi{cluster_idx}{stim_idx}(:, max_ampl_window)', [], 1);
    end
end

% group and get mean across mice
mean_max_ampl_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        mean_max_ampl_kernel_roi{cluster_idx}{stim_idx} = ap.groupfun(@nanmean, ...
            all_max_ampl_kernel_roi{cluster_idx}{stim_idx}(use_days)', wf_animal_group_clusters_indices);
    end
end

median_max_ampl_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        median_max_ampl_kernel_roi{cluster_idx}{stim_idx} = ap.groupfun(@median, ...
            all_max_ampl_kernel_roi{cluster_idx}{stim_idx}(use_days)', wf_animal_group_clusters_indices);
    end
end

% do sem for errorbar
std_max_ampl_kernel_roi = cell(num_clusters, 1);
sem_max_ampl_kernel_roi = cell(num_clusters, 1);
for cluster_idx=1:num_clusters
    for stim_idx = 1:numel(unique_stims)
        std_max_ampl_kernel_roi{cluster_idx}{stim_idx} = ap.groupfun(@nanstd, ...
            all_max_ampl_kernel_roi{cluster_idx}{stim_idx}(use_days)', wf_animal_group_clusters_indices);
        sem_max_ampl_kernel_roi{cluster_idx}{stim_idx} = std_max_ampl_kernel_roi{cluster_idx}{stim_idx} ./ sqrt(num_animals_stim_wf{stim_idx});
    end
end

%% plot max amplitude for manual kernel ROIs
days_for_plot = -3:2;
curr_color = 'k';
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

        for_plot_mean_max_pfc_roi = mean_max_ampl_kernel_roi{cluster_idx}{stim_idx};
        for_plot_sem_max_pfc_roi = sem_max_ampl_kernel_roi{cluster_idx}{stim_idx};

        plotted_days = these_days_from_learning(plot_day_idx);

        %     % get num animals for legend
        %     num_animals_plotted = num_animals_stim_wf{stim_idx}(plot_day_idx);
        %
        %     % make legend
        %     legend_for_plot = arrayfun(@(day, num) ['Day ' num2str(day) ' (n = ' num2str(num) ')'], ...
        %         plotted_days, num_animals_plotted, 'UniformOutput', false);

        nexttile
        errorbar(plotted_days, for_plot_mean_max_pfc_roi(plot_day_idx), for_plot_sem_max_pfc_roi(plot_day_idx), '-o', 'CapSize', 0, ...
            'MarkerFaceColor', curr_color, 'MarkerEdgeColor', curr_color, 'Color', curr_color);
        title(['Stim ' num2str(unique_stims(stim_idx))])
    end
    sgtitle(['Max amplitude for cluster ' num2str(cluster_idx)])
end


