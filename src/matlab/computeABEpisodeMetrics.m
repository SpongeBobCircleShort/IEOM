function metrics = computeABEpisodeMetrics(frame_log, schedule, config)
% computeABEpisodeMetrics: Reduce one episode frame log into summary metrics.

    completion_time_sec = frame_log(end).time_sec + config.dt_sec;
    human_states = {frame_log.human_state};
    robot_modes = {frame_log.robot_mode};
    robot_speed_scale = [frame_log.robot_speed_scale];
    simultaneous_shared = [frame_log.simultaneous_shared_zone];
    overlap_events = countTransitions(simultaneous_shared);
    unsafe_events = countTransitions([frame_log.emergency_active]);
    hesitation_events_from_states = countHesitationOnsets(human_states);
    hesitation_events = hesitation_events_from_states;
    hesitation_events_from_flags = NaN;
    if isfield(frame_log, 'hesitation_onset')
        hesitation_events_from_flags = sum([frame_log.hesitation_onset] > 0.5);
        hesitation_events = hesitation_events_from_flags;
    end

    metrics = struct( ...
        'scenario_name', schedule.scenario_name, ...
        'seed', schedule.seed, ...
        'policy_name', '', ...
        'backend', '', ...
        'success', double(strcmp(frame_log(end).human_state, 'done')), ...
        'completion_time_sec', completion_time_sec, ...
        'min_separation_m', min([frame_log.separation_m]), ...
        'overlap_event_count', overlap_events, ...
        'overlap_frames', sum(simultaneous_shared), ...
        'unsafe_close_call_count', unsafe_events, ...
        'robot_wait_time_sec', sum(strcmp(robot_modes, 'hold') | strcmp(robot_modes, 'emergency_stop')) * config.dt_sec, ...
        'human_wait_time_sec', sum([frame_log.human_speed] < 1e-6) * config.dt_sec, ...
        'robot_hold_count', countTransitions(strcmp(robot_modes, 'hold')), ...
        'robot_slow_frames', sum(strcmp(robot_modes, 'slow')), ...
        'policy_intervention_frames', sum(strcmp(robot_modes, 'hold') | strcmp(robot_modes, 'slow')), ...
        'human_rework_count', max([frame_log.retry_count]), ...
        'mild_hesitation_time_sec', sum(strcmp(human_states, 'mild_hesitation')) * config.dt_sec, ...
        'strong_hesitation_time_sec', sum(strcmp(human_states, 'strong_hesitation')) * config.dt_sec, ...
        'hesitation_event_count', hesitation_events, ...
        'hesitation_event_count_from_states', hesitation_events_from_states, ...
        'hesitation_event_count_consistent', double(isnan(hesitation_events_from_flags) || hesitation_events_from_flags == hesitation_events_from_states), ...
        'response_latency_sec', NaN, ...
        'mean_robot_speed_scale', mean(robot_speed_scale), ...
        'shared_zone_simultaneous_frames', sum(simultaneous_shared));
end

function count = countHesitationOnsets(human_states)
    if isempty(human_states)
        count = 0;
        return;
    end

    hesitation_states = {'mild_hesitation', 'strong_hesitation', 'correction_rework'};
    count = 0;
    previous_hesitating = false;

    for idx = 1:numel(human_states)
        current_hesitating = any(strcmp(human_states{idx}, hesitation_states));
        if current_hesitating && ~previous_hesitating
            count = count + 1;
        end
        previous_hesitating = current_hesitating;
    end
end

function count = countTransitions(flag_values)
    if isempty(flag_values)
        count = 0;
        return;
    end

    count = 0;
    previous = false;
    for idx = 1:numel(flag_values)
        current = logical(flag_values(idx));
        if current && ~previous
            count = count + 1;
        end
        previous = current;
    end
end
