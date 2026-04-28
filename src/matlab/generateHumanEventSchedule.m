function schedule = generateHumanEventSchedule(scenario_name, seed, config)
% generateHumanEventSchedule: Create deterministic human-event schedule for one scenario/seed.

    scenarios = buildABScenarioLibrary();
    scenario = findScenario(scenarios, scenario_name);

    rng(seed, 'twister');

    hesitation_progress = interpolateRange(scenario.hesitation_progress_range, rand());
    hesitation_duration = round(interpolateRange(scenario.hesitation_duration_range, rand()));
    rework_progress = interpolateRange(scenario.rework_progress_range, rand());

    if hesitation_duration <= 0
        hesitation_onset_frame = 0;
        hesitation_end_frame = 0;
    else
        hesitation_onset_frame = max(1, round(hesitation_progress * config.max_frames));
        hesitation_end_frame = min(config.max_frames, hesitation_onset_frame + hesitation_duration);
    end
    rework_trigger_frame = 0;
    if rework_progress > 0.0
        rework_trigger_frame = max(1, round(rework_progress * config.max_frames));
    end

    if scenario.escalation && hesitation_duration > 0
        strong_onset_frame = hesitation_onset_frame + max(4, floor(0.4 * hesitation_duration));
        strong_onset_frame = min(strong_onset_frame, hesitation_end_frame);
    else
        strong_onset_frame = 0;
    end

    schedule = struct( ...
        'scenario_name', scenario.name, ...
        'seed', seed, ...
        'hesitation_onset_frame', hesitation_onset_frame, ...
        'hesitation_duration_frames', hesitation_duration, ...
        'hesitation_end_frame', hesitation_end_frame, ...
        'escalate_to_strong', logical(scenario.escalation), ...
        'strong_onset_frame', strong_onset_frame, ...
        'rework_trigger_frame', rework_trigger_frame, ...
        'human_nominal_speed_mps', interpolateRange(scenario.human_speed_range, rand()), ...
        'robot_nominal_speed_mps', interpolateRange(scenario.robot_speed_range, rand()), ...
        'jitter_scale_m', scenario.jitter_scale, ...
        'crowding_sensitivity_threshold_m', scenario.crowding_sensitivity_threshold_m, ...
        'early_robot_release_frames', scenario.early_robot_release_frames, ...
        'robot_release_frame', max(1, scenario.early_robot_release_frames) ...
    );
end

function scenario = findScenario(scenarios, scenario_name)
    scenario = struct();
    for idx = 1:numel(scenarios)
        if strcmp(scenarios(idx).name, scenario_name)
            scenario = scenarios(idx);
            return;
        end
    end
    error('Unknown scenario: %s', scenario_name);
end

function value = interpolateRange(range_values, ratio)
    if numel(range_values) ~= 2
        error('Range must contain exactly two values.');
    end
    value = range_values(1) + (range_values(2) - range_values(1)) * ratio;
end
