save_path = '\\qnap-ap001.dpag.ox.ac.uk\APlab\Users\Andrada-Maria_Marica\long_str_ctx_data';

animals = { ...
    'AM011','AM012','AM014','AM015','AM016','AM017', ...
    'AM018','AM019','AM021','AM022','AM026','AM029', ...
    'AP023','AP025'};

workflow = {'stim_wheel_right*'};

all_bhv_cell = cell(length(animals), 1);

for animal_idx=1:length(animals)
    animal = animals{animal_idx};
    recordings = plab.find_recordings(animal, [], workflow);
    
    bhv_animal = table;
    learned_days = nan(length(recordings), 1);
    days_from_learning = nan(length(recordings), 1); 
    for use_rec=1:length(recordings)

        rec_day = recordings(use_rec).day;
        rec_time = recordings(use_rec).recording{end};
        verbose = true;
        load_parts.behavior = true;
        ap.load_recording

        [stimwheel_pval,...
            stimwheel_rxn_mean, ...
            stimwheel_rxn_null_mean] = ...
            AP_stimwheel_association_pvalue(stimOn_times,trial_events,stim_to_move, 'mean');

        learned_days(use_rec) = stimwheel_pval < 0.05;

        if use_rec==1
            learned_days(use_rec) = 0;
        end
        
        % get good reward times
        trial_outcome = logical([trial_events.values.Outcome]);
        stim_to_reward = reward_times_task - stimOn_times(trial_outcome);

        % save
        bhv_animal.animal(use_rec) = {animal};
        bhv_animal.rec_day(use_rec) = {rec_day};
        bhv_animal.stimwheel_pval(use_rec) = {stimwheel_pval};
        bhv_animal.stimwheel_rxn_mean(use_rec) = {stimwheel_rxn_mean};
        bhv_animal.stimwheel_rxn_null_mean(use_rec) = {stimwheel_rxn_null_mean};
        % reaction times
        bhv_animal.stim_to_move(use_rec) = {stim_to_move(1:n_trials)};
        bhv_animal.trial_outcome(use_rec) = {trial_outcome(1:n_trials)};
        bhv_animal.stim_to_reward(use_rec) = {stim_to_reward(1:n_trials)};

    end

    learned_day = nanmax([nan, find(learned_days, 1, 'first')]);
    recording_days = 1:length(recordings);
    days_from_learning = recording_days - learned_day;

    bhv_animal.learned_days = learned_days;
    bhv_animal.days_from_learning = days_from_learning';

    all_bhv_cell{animal_idx} = bhv_animal;

end

bhv = vertcat(all_bhv_cell{:});

figure;
for animal_idx=1:length(animals)
    animal = animals(animal_idx);
    this_stim_wheel_pval = cell2mat(bhv.stimwheel_pval(strcmp(bhv.animal, animal)));
    plot(this_stim_wheel_pval);
    hold on
end
legend(animals)

save_name = fullfile(save_path, 'swr_bhv');
save(save_name, "bhv", "-v7.3");

clear all;