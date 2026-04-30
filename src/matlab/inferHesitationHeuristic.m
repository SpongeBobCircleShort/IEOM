function prediction = inferHesitationHeuristic(features, config)
% inferHesitationHeuristic: Deterministic inference stub with Python-CLI-compatible schema.

    phase = detectTaskPhase(features, config);
    phase_weight = phaseHesitationWeight(phase);
    weighted_pause_ratio = features.pause_ratio * phase_weight;
    ambiguity_level = max(config.env_context.occlusion_level, config.env_context.ghost_proximity_level);
    ambiguity_dampen = max(0.35, 1.0 - 0.55 * ambiguity_level);
    weighted_pause_ratio = weighted_pause_ratio * ambiguity_dampen;
    expected_pause_ratio = min(0.95, max(0.05, config.env_context.expected_pause_duration_sec / ...
        max(0.1, (config.window_size_frames / config.frame_rate_hz))));

    state = 'normal_progress';
    if double(features.retry_count) > 0 && features.progress_delta < config.heuristic.rework_progress_delta_threshold
        state = 'correction_rework';
    elseif weighted_pause_ratio >= max(config.heuristic.pause_ratio_strong, expected_pause_ratio * 0.9) || ...
            features.mean_hand_speed < config.heuristic.mean_speed_strong
        state = 'strong_hesitation';
    elseif weighted_pause_ratio >= max(config.heuristic.pause_ratio_mild, expected_pause_ratio * 0.6) || ...
            features.mean_hand_speed < config.heuristic.mean_speed_mild || ...
            double(features.reversal_count) >= config.heuristic.reversal_mild_count || ...
            features.human_robot_distance < config.heuristic.distance_mild
        state = 'mild_hesitation';
    end

    probabilities = buildProbabilities(state);
    if ambiguity_level >= 0.60
        probabilities = flattenProbabilities(probabilities, 0.20 * ambiguity_level);
    end
    prediction = struct( ...
        'state', state, ...
        'state_probabilities', probabilities, ...
        'future_hesitation_prob', futureHesitationProbability(state), ...
        'future_correction_prob', futureCorrectionProbability(state), ...
        'confidence', probabilities.(state), ...
        'task_phase', phase, ...
        'ambiguity_level', ambiguity_level, ...
        'window_size_frames', config.window_size_frames, ...
        'frame_rate_hz', config.frame_rate_hz);
end

function phase = detectTaskPhase(features, config)
    if config.env_context.task_precision_required >= 0.7 && features.mean_hand_speed < 0.05
        phase = 'PRECISION_WORK';
    elseif features.human_robot_distance < 0.18 && features.progress_delta > -0.01
        phase = 'HANDOFF_PREP';
    elseif features.progress_delta < -0.02 || double(features.retry_count) > 0
        phase = 'INSPECTION';
    elseif features.mean_hand_speed < 0.04 && features.human_robot_distance > 0.22
        phase = 'RETREAT';
    else
        phase = 'APPROACH';
    end
end

function probabilities = flattenProbabilities(probabilities, alpha)
    keys = fieldnames(probabilities);
    n = numel(keys);
    for idx = 1:n
        key = keys{idx};
        value = probabilities.(key);
        probabilities.(key) = (1 - alpha) * value + alpha * (1.0 / n);
    end
end

function weight = phaseHesitationWeight(phase)
    switch phase
        case 'PRECISION_WORK'
            weight = 0.10;
        case 'INSPECTION'
            weight = 0.20;
        case 'HANDOFF_PREP'
            weight = 0.90;
        case 'RETREAT'
            weight = 0.35;
        otherwise
            weight = 1.00;
    end
end

function probabilities = buildProbabilities(state)
    switch state
        case 'normal_progress'
            probabilities = struct( ...
                'normal_progress', 0.88, ...
                'mild_hesitation', 0.06, ...
                'strong_hesitation', 0.03, ...
                'correction_rework', 0.03, ...
                'ready_for_robot_action', 0.00, ...
                'overlap_risk', 0.00);
        case 'mild_hesitation'
            probabilities = struct( ...
                'normal_progress', 0.08, ...
                'mild_hesitation', 0.78, ...
                'strong_hesitation', 0.08, ...
                'correction_rework', 0.06, ...
                'ready_for_robot_action', 0.00, ...
                'overlap_risk', 0.00);
        case 'strong_hesitation'
            probabilities = struct( ...
                'normal_progress', 0.03, ...
                'mild_hesitation', 0.12, ...
                'strong_hesitation', 0.80, ...
                'correction_rework', 0.05, ...
                'ready_for_robot_action', 0.00, ...
                'overlap_risk', 0.00);
        case 'correction_rework'
            probabilities = struct( ...
                'normal_progress', 0.02, ...
                'mild_hesitation', 0.08, ...
                'strong_hesitation', 0.04, ...
                'correction_rework', 0.86, ...
                'ready_for_robot_action', 0.00, ...
                'overlap_risk', 0.00);
        otherwise
            probabilities = struct( ...
                'normal_progress', 0.88, ...
                'mild_hesitation', 0.06, ...
                'strong_hesitation', 0.03, ...
                'correction_rework', 0.03, ...
                'ready_for_robot_action', 0.00, ...
                'overlap_risk', 0.00);
    end
end

function value = futureHesitationProbability(state)
    switch state
        case 'normal_progress'
            value = 0.12;
        case 'mild_hesitation'
            value = 0.58;
        case 'strong_hesitation'
            value = 0.86;
        case 'correction_rework'
            value = 0.72;
        otherwise
            value = 0.20;
    end
end

function value = futureCorrectionProbability(state)
    switch state
        case 'correction_rework'
            value = 0.76;
        case 'strong_hesitation'
            value = 0.34;
        case 'mild_hesitation'
            value = 0.18;
        otherwise
            value = 0.08;
    end
end
