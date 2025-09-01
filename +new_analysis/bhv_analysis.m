%% BEHAVIOUR
%% load
save_path = '\\qnap-ap001.dpag.ox.ac.uk\APlab\Users\Andrada-Maria_Marica\long_str_ctx_data';
load(fullfile(save_path, 'swr_bhv.mat'));
%% - wheel trace with stim onset and move onset + heatmaps for RT
animal = 'AM021';

animal_idx = strcmp(bhv.animal, animal);
days_from_learning = bhv.days_from_learning(animal_idx);
stimwheel_pval = bhv.stimwheel_pval(animal_idx);
% bhv_days = bhv.bhv_days{animal_idx};

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
saveas(gcf,'D:\corel_test\figure1a.svg')

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

% Find the imagesc object
hImage = findobj(gca,'Type','Image');
% Hide the image only (heatmap), leave axes, ticks, labels, colorbar, title visible
set(hImage,'Visible','off');
% Export SVG with painters (vector elements only)
print(gcf, '-dsvg', '-painters', 'D:\corel_test\figure1b_overlay.svg');
% Turn the image back on if needed
set(hImage,'Visible','on');

% Hide text/axes temporarily for clean raster
set(gca,'XTickLabel',[],'YTickLabel',[]);
xlabel(''); ylabel('');
title('');
print(gcf, '-dpng', '-r600', 'D:\corel_test\figure1b_raster.png')
% Restore text/labels for display
xlabel('Time from stim onset (100ms)', 'FontSize', 38)
ylabel('Trial sorted by RT', 'FontSize', 38)
title('Day 2', 'FontSize', 18)

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

all_days_from_learning = bhv.days_from_learning;
all_rxn_null_mean = bhv.stimwheel_rxn_null_mean;
all_rxn_mean = bhv.stimwheel_rxn_mean;

unique_animals = unique(bhv.animal);
unique_days_from_learning = unique(all_days_from_learning);          

% Step 2: Initialize output cell array (11 x num_unique_days)
aligned_all_rxn_null_mean = nan(length(unique_animals), numel(unique_days_from_learning));
aligned_all_rxn_mean = nan(length(unique_animals), numel(unique_days_from_learning));

% Step 3: Fill in output_matrix based on learning days
for animal_id = 1:length(unique_animals)  % Loop through each animal
    this_animal_idx = strcmp(bhv.animal, unique_animals(animal_id));
    current_days = all_days_from_learning(this_animal_idx);            
    this_rxn_null_mean = all_rxn_null_mean(this_animal_idx);
    this_rxn_mean = all_rxn_mean(this_animal_idx);
    % Loop through each day for the current animal
    for day_idx = 1:length(current_days)
        day = current_days(day_idx);                 
        new_day_index = find(unique_days_from_learning == day);   
        
        % Store the data for this day in the corresponding position
        if ~isempty(this_rxn_null_mean{day_idx})
            aligned_all_rxn_null_mean(animal_id, new_day_index) = this_rxn_null_mean{day_idx};
            aligned_all_rxn_mean(animal_id, new_day_index) = this_rxn_mean{day_idx};
        end
    end
end

% choose interval
good_days = unique_days_from_learning<=3 & unique_days_from_learning>=-3;

% get avg
mean_aligned_all_rxn_mean = nanmean(aligned_all_rxn_mean(:, good_days), 1);
mean_aligned_all_rxn_null_mean = nanmean(aligned_all_rxn_null_mean(:, good_days), 1);

% Calculate SEM
sem_aligned_all_rxn_mean = nanstd(aligned_all_rxn_mean(:, good_days), 0, 1) ./ sqrt(sum(~isnan(aligned_all_rxn_mean(:, good_days)), 1));
sem_aligned_all_rxn_null_mean = nanstd(aligned_all_rxn_null_mean(:, good_days), 0, 1) ./ sqrt(sum(~isnan(aligned_all_rxn_null_mean(:, good_days)), 1));

% do index: diff over sum
performance_index = (aligned_all_rxn_null_mean(:, good_days) - aligned_all_rxn_mean(:, good_days))./ ...
    (aligned_all_rxn_mean(:, good_days)+aligned_all_rxn_null_mean(:, good_days));
mean_performance_index = nanmean(performance_index, 1);
sem_performance_index = nanstd(performance_index, 0, 1) ./ sqrt(sum(~isnan(performance_index), 1));

% figure for real vs null
figure('Position', [680   400   690   570]);
hold on;
% Plot mean for reaction med with error bars
errorbar(unique_days_from_learning(good_days), mean_aligned_all_rxn_mean, sem_aligned_all_rxn_mean, ...
    'DisplayName', 'Mean Reaction Med', 'Color', 'k', 'CapSize', 0);
set(gca, 'YScale', 'log')
% Create shaded area for SEM of null reaction mean
x = unique_days_from_learning(good_days);               % [1 x N]
y_mean = mean_aligned_all_rxn_null_mean;                % [1 x N]
y_sem = sem_aligned_all_rxn_null_mean;                  % [1 x N]
x = x(:)';
y_mean = y_mean(:)';
y_sem = y_sem(:)';
x_fill = [x, fliplr(x)];                                % 1 x (2N)
y_fill = [y_mean + y_sem, fliplr(y_mean - y_sem)];      % 1 x (2N)
fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
set(gca, 'YScale', 'log')
% Plot mean for null reaction med
plot(unique_days_from_learning(good_days), mean_aligned_all_rxn_null_mean, 'DisplayName', 'Mean Null Reaction Med', 'Color', [0.5 0.5 0.5]);
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
errorbar(unique_days_from_learning(good_days), mean_performance_index, sem_performance_index, 'DisplayName', 'Mean Performance Index', ...
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
animal_idx = strcmp(unique_animals, animal);
% figure for real vs null
figure('Position',   [370   180   700  580]);
hold on;
% Plot mean for reaction med with error bars
plot(unique_days_from_learning(good_days), aligned_all_rxn_mean(animal_idx, good_days), 'Color', 'k');
set(gca, 'YScale', 'log')
% Plot mean for null reaction med
plot(unique_days_from_learning(good_days), aligned_all_rxn_null_mean(animal_idx, good_days), 'Color', [0.5 0.5 0.5]);
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

%% get min mouse ld
per_mouse_days_from_learning = cell(length(unique_animals) , 1);
min_mouse_ld = nan(length(unique_animals) , 1);
for animal_idx=1:length(unique_animals) 
    per_mouse_days_from_learning{animal_idx} = bhv.days_from_learning(strcmp(bhv.animal, unique_animals(animal_idx)));
    min_mouse_ld(animal_idx) = min(per_mouse_days_from_learning{animal_idx});
end

% plot
figure('Position',   [370   180   700  580]);
hold on;
histogram(abs(min_mouse_ld), 'FaceColor', [0.5 0.5 0.5])
xlabel('Days to learn association', 'FontSize', 45);
ylabel('# of mice', 'FontSize', 45);
ax = get(gca);
% Customize tick labels and spacing
ax.XAxis.FontSize = 20;  % Set X-axis tick label font size
ax.YAxis.FontSize = 20;  % Set Y-axis tick label font size
