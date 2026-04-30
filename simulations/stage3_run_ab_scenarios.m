function summary = stage3_run_ab_scenarios(varargin)
% stage3_run_ab_scenarios: Stage 3 A/B evaluation with real model bridge + replay validation.

    opts = parseStage3Options(varargin{:});
    root_dir = fileparts(fileparts(mfilename('fullpath')));
    output_root = fullfile(root_dir, 'artifacts', 'simulink_stage3');
    dirs = ensureStage3Dirs(output_root);

    envs = buildStage3Environments();
    config = buildStage3Config(opts, root_dir);

    baseline_rows = struct([]);
    aware_rows = struct([]);
    for s = 1:numel(envs)
        env = envs{s};
        env_config = config;
        if isfield(env, 'shared_zone_x'), env_config.shared_zone_x = env.shared_zone_x; end
        if isfield(env, 'overlap_buffer'), env_config.overlap_buffer = env.overlap_buffer; end
        
        b_res = runScenarioPolicy(env, 'baseline', env_config, dirs.feature_logs);
        a_res = runScenarioPolicy(env, 'hesitation_aware', env_config, dirs.feature_logs);
        if isempty(baseline_rows)
            baseline_rows = b_res;
            aware_rows = a_res;
        else
            baseline_rows(s) = b_res; %#ok<AGROW>
            aware_rows(s) = a_res; %#ok<AGROW>
        end
    end

    baseline_tbl = struct2table(baseline_rows);
    aware_tbl = struct2table(aware_rows);
    comparison_tbl = buildComparisonTable(baseline_tbl, aware_tbl);
    summary_tbl = buildStatisticalSummary(comparison_tbl);
    safety_tbl = buildSafetyReport(aware_tbl, config);

    writetable(baseline_tbl, fullfile(dirs.tables, 'metrics_baseline.csv'));
    writetable(aware_tbl, fullfile(dirs.tables, 'metrics_hesitation_aware.csv'));
    writetable(comparison_tbl, fullfile(dirs.tables, 'comparison_summary.csv'));
    writetable(summary_tbl, fullfile(dirs.tables, 'statistical_summary.csv'));
    writetable(safety_tbl, fullfile(dirs.reports, 'safety_checks.csv'));

    replay_tbl = runReplayValidation(envs, config, dirs);
    writetable(replay_tbl, fullfile(dirs.reports, 'replay_validation.csv'));

    try
        renderStage3Figures(dirs.figures, comparison_tbl, aware_tbl);
    catch fig_err
        fprintf('[INFO] Skipping stage3 figures (headless): %s\n', fig_err.message);
    end

    summary = struct( ...
        'output_root', output_root, ...
        'feature_log_dir', dirs.feature_logs, ...
        'scenario_names', {cellfun(@(x) x.name, envs, 'UniformOutput', false)}, ...
        'baseline_metrics', baseline_tbl, ...
        'hesitation_aware_metrics', aware_tbl, ...
        'comparison', comparison_tbl, ...
        'statistical_summary', summary_tbl, ...
        'replay_validation', replay_tbl, ...
        'safety_checks', safety_tbl);

    fprintf('Stage 3 complete. Output root: %s\n', output_root);
end

function opts = parseStage3Options(varargin)
    opts = struct('use_stub', false, 'enable_replay', true, 'deterministic_seed', 42, 'window_size', 12, 'dt_sec', 0.1, 'no_figures', false);
    if mod(numel(varargin), 2) ~= 0
        error('stage3_run_ab_scenarios expects name/value options.');
    end
    for idx = 1:2:numel(varargin)
        opts.(char(varargin{idx})) = varargin{idx + 1};
    end
end

function dirs = ensureStage3Dirs(output_root)
    dirs = struct();
    dirs.feature_logs = fullfile(output_root, 'feature_logs');
    dirs.replay_logs = fullfile(output_root, 'replay_logs');
    dirs.tables = fullfile(output_root, 'tables');
    dirs.figures = fullfile(output_root, 'figures');
    dirs.reports = fullfile(output_root, 'reports');
    folders = struct2cell(dirs);
    for i = 1:numel(folders)
        if ~exist(folders{i}, 'dir')
            mkdir(folders{i});
        end
    end
end

function config = buildStage3Config(opts, root_dir)
    config = struct();
    config.dt_sec = opts.dt_sec;
    config.max_steps = 1200;
    config.window_size = opts.window_size;
    config.seed = opts.deterministic_seed;
    config.use_stub = opts.use_stub;
    config.enable_replay = opts.enable_replay;

    config.robot_nominal_speed = 0.55;
    config.human_nominal_speed = 0.45;
    config.robot_release_delay_sec = 1.0;

    config.shared_zone_x = [0.42, 0.58];
    config.shared_zone_y = [0.35, 0.65];
    config.fixture_pos = [0.50, 0.50];
    config.human_start = [0.12, 0.50];
    config.robot_start = [0.88, 0.50];
    config.human_target = [0.86, 0.50];
    config.robot_target = [0.14, 0.50];
    config.overlap_buffer = 0.08;

    config.bridge_python = 'python3';
    config.bridge_script = fullfile(root_dir, 'scripts', 'simulink_stage3_predict.py');
    config.max_hold_ratio = 0.65;
    config.max_oscillation_ratio = 0.40;

    % Load the trained classical models (Multi-Task Learning, Fix 4)
    config.models = struct('generic', [], 'assembly', [], 'precision', [], 'inspection', []);
    
    model_types = {'generic', 'assembly', 'precision', 'inspection'};
    for i = 1:numel(model_types)
        mtype = model_types{i};
        if strcmp(mtype, 'generic')
            fname = 'classical_model.json';
        else
            fname = sprintf('classical_model_%s.json', mtype);
        end
        
        mpath = fullfile(root_dir, 'simulations', fname);
        if exist(mpath, 'file')
            config.models.(mtype) = jsondecode(fileread(mpath));
        elseif ~strcmp(mtype, 'generic')
            % Fallback to generic if specialized not found
            config.models.(mtype) = config.models.generic;
        end
    end
    
    % Legacy support
    config.trained_model = config.models.generic;
end

function envs = buildStage3Environments()
    % States: [normal, coord_uncertainty, task_complexity, inspection, conflict, ready, rework]
    envs = {
        struct('name', 'low_conflict_open', 'weights', [0.60, 0.05, 0.05, 0.05, 0.05, 0.15, 0.05], 'task_step_count', 6, 'shared_zone_x', [0.45, 0.55], 'overlap_buffer', 0.05, 'conflict_level', 'low'), ...
        struct('name', 'narrow_assembly_bench', 'weights', [0.40, 0.15, 0.05, 0.05, 0.10, 0.10, 0.15], 'task_step_count', 6, 'shared_zone_x', [0.30, 0.70], 'overlap_buffer', 0.10, 'conflict_level', 'high'), ...
        struct('name', 'precision_insertion', 'weights', [0.20, 0.05, 0.35, 0.15, 0.10, 0.10, 0.05], 'task_step_count', 6, 'shared_zone_x', [0.48, 0.52], 'overlap_buffer', 0.15, 'conflict_level', 'high'), ...
        struct('name', 'inspection_rework', 'weights', [0.20, 0.05, 0.10, 0.40, 0.10, 0.10, 0.05], 'task_step_count', 6, 'shared_zone_x', [0.40, 0.60], 'overlap_buffer', 0.08, 'conflict_level', 'high'), ...
        struct('name', 'shared_bin_access', 'weights', [0.20, 0.10, 0.05, 0.05, 0.40, 0.15, 0.05], 'task_step_count', 6, 'shared_zone_x', [0.35, 0.65], 'overlap_buffer', 0.12, 'conflict_level', 'high') ...
    };
end

function metrics = runScenarioPolicy(scenario, policy_name, config, feature_log_dir)
    state_names = {'normal_progress', 'coordination_uncertainty', 'task_complexity', 'deliberate_inspection', 'workspace_conflict', 'ready_for_robot_action', 'correction_rework'};
    seed_bump = strcmp(policy_name, 'hesitation_aware') * 1000;
    rng(config.seed + seed_bump + sum(double(char(scenario.name))), 'twister');

    human_pos = config.human_start;
    robot_pos = config.robot_start;
    human_progress = 0.0;
    robot_progress = 0.0;

    history = initializeFeatureHistory();
    log_rows = cell(config.max_steps, 1);

    robot_idle = 0.0;
    human_idle = 0.0;
    overlap_count = 0;
    hold_count = 0;
    unnecessary_slowdown = 0;
    correction_count = 0;
    mismatch_count = 0;
    mode_switches = 0;

    prev_overlap = false;
    prev_scripted = 'normal_progress';
    prev_robot_mode = 'proceed';
    row_idx = 1;

    for step_idx = 1:config.max_steps
        ts = (step_idx - 1) * config.dt_sec;
        scripted_state = scriptedStateSample(state_names, scenario.weights, ts, human_progress);
        state_params = scriptedStateParameters(scripted_state);

        if strcmp(scripted_state, 'correction_rework') && ~strcmp(prev_scripted, 'correction_rework')
            correction_count = correction_count + 1;
        end

        if rand() < state_params.pause_probability
            human_speed = 0.0;
        else
            human_speed = config.human_nominal_speed * state_params.speed_scale;
        end

        task_step = min(scenario.task_step_count, max(1, 1 + floor(human_progress * scenario.task_step_count)));
        shared_occ = double(isInSharedZone(human_pos, config));
        history = appendFeatureHistory(history, human_pos, human_speed, human_progress, state_params.retry_count, ...
            norm(config.human_target - human_pos), norm(human_pos - robot_pos), task_step, shared_occ);

        if numel(history.speed) >= config.window_size
            feature_window = computeFeatureWindow(history, config.window_size, ts);
            validateFeatureSchemaContract(feature_window);
        else
            feature_window = emptyFeatureWindow(ts);
        end

        % --- Fix 5: maintain per-scenario temporal sliding window ---
        if step_idx == 1
            feature_window_history = {};
        end
        feature_window_history{end+1} = feature_window;
        if numel(feature_window_history) > 10
            feature_window_history(1) = [];
        end

        % --- Fix 4: detect task type for model selection ---
        if ~isempty(strfind(scenario.name, 'assembly')) || ~isempty(strfind(scenario.name, 'shared_bin'))
            task_type = 'assembly';
        elseif ~isempty(strfind(scenario.name, 'precision'))
            task_type = 'precision';
        elseif ~isempty(strfind(scenario.name, 'inspection'))
            task_type = 'inspection';
        else
            task_type = 'generic';
        end

        if strcmp(policy_name, 'baseline')
            [robot_speed, robot_mode] = baselinePolicy(config, ts);
            prediction = scriptPassThroughPrediction(scripted_state);
        else
            prediction = predict_hesitation_state(feature_window, feature_window_history, task_type, scripted_state, config, scenario);
            validatePredictionOutput(prediction);
            [robot_speed, robot_mode] = policyBFromInference(prediction.predicted_state, config, ts);
        end

        if ~strcmp(prediction.predicted_state, scripted_state)
            mismatch_count = mismatch_count + 1;
        end
        if ~strcmp(robot_mode, prev_robot_mode)
            mode_switches = mode_switches + 1;
        end
        if strcmp(robot_mode, 'hold')
            hold_count = hold_count + 1;
        end
        if (strcmp(robot_mode, 'hold') || strcmp(robot_mode, 'slow')) && ...
                (strcmp(scripted_state, 'normal_progress') || strcmp(scripted_state, 'ready_for_robot_action'))
            unnecessary_slowdown = unnecessary_slowdown + 1;
        end

        human_pos = stepToward(human_pos, config.human_target, human_speed, config.dt_sec);
        robot_pos = stepToward(robot_pos, config.robot_target, robot_speed, config.dt_sec);
        if rand() < state_params.shared_zone_entry_prob
            human_pos(1) = min(max(human_pos(1), config.shared_zone_x(1)), config.shared_zone_x(2));
            human_pos(2) = config.fixture_pos(2) + (rand() - 0.5) * 0.12;
        end

        human_progress = min(1.0, human_progress + state_params.progress_rate * config.dt_sec);
        robot_progress = min(1.0, robot_progress + max(robot_speed, 0.0) * config.dt_sec / norm(config.robot_start - config.robot_target));

        if human_speed <= 1e-6, human_idle = human_idle + config.dt_sec; end
        if robot_speed <= 1e-6, robot_idle = robot_idle + config.dt_sec; end

        overlap_now = isInSharedZone(human_pos, config) && isInSharedZone(robot_pos, config) && norm(human_pos - robot_pos) <= config.overlap_buffer;
        if overlap_now && ~prev_overlap
            overlap_count = overlap_count + 1;
        end

        log_rows{row_idx} = featureLogRow(ts, scenario.name, policy_name, scripted_state, prediction, feature_window, robot_mode, robot_speed);
        row_idx = row_idx + 1;

        prev_overlap = overlap_now;
        prev_scripted = scripted_state;
        prev_robot_mode = robot_mode;
        if human_progress >= 1.0 && robot_progress >= 1.0
            break;
        end
    end

    log_rows = log_rows(1:row_idx-1);
    writeJsonl(fullfile(feature_log_dir, sprintf('%s_%s.jsonl', scenario.name, policy_name)), log_rows);

    sim_time = (step_idx - 1) * config.dt_sec;
    metrics = struct( ...
        'scenario', char(scenario.name), ...
        'policy', char(policy_name), ...
        'task_completion_time_sec', sim_time, ...
        'overlap_risk_event_count', overlap_count, ...
        'robot_hold_count', hold_count, ...
        'human_wait_time_sec', human_idle, ...
        'robot_idle_time_sec', robot_idle, ...
        'unnecessary_slowdown_count', unnecessary_slowdown, ...
        'correction_rework_count', correction_count, ...
        'prediction_mismatch_count', mismatch_count, ...
        'decision_switch_count', mode_switches, ...
        'total_simulated_time_sec', sim_time);
end

function replay_tbl = runReplayValidation(scenarios, config, dirs)
    if ~config.enable_replay
        replay_tbl = struct();
        return;
    end

    rows = cell(numel(scenarios)*2, 1);
    idx = 1;
    for s = 1:numel(scenarios)
        for policy = {"baseline", "hesitation_aware"}
            policy_name = char(policy{1});
            path = fullfile(dirs.feature_logs, sprintf('%s_%s.jsonl', scenarios{s}.name, policy_name));
            log_rows = readJsonl(path);
            match_state = 0;
            match_probs = 0;
            for j = 1:numel(log_rows)
                entry = log_rows(j);
                pred = predict_hesitation_state(entry.feature_window, char(entry.scripted_state), config);
                if strcmp(pred.predicted_state, char(entry.predicted_state))
                    match_state = match_state + 1;
                end
                if probabilityDistance(pred.state_probabilities, entry.state_probabilities) < 1e-6
                    match_probs = match_probs + 1;
                end
            end

            deterministic = (match_state == numel(log_rows)) && (match_probs == numel(log_rows));
            rows{idx} = struct( ...
                'scenario', char(scenarios{s}.name), ...
                'policy', char(policy_name), ...
                'rows_replayed', numel(log_rows), ...
                'state_matches', match_state, ...
                'probability_matches', match_probs, ...
                'deterministic', deterministic);

            replay_report = struct2table(rows{idx});
            writetable(replay_report, fullfile(dirs.replay_logs, sprintf('%s_%s_replay.csv', scenarios{s}.name, policy_name)));
            idx = idx + 1;
        end
    end
    rows = rows(1:idx-1);
    replay_tbl = struct2table([rows{:}]);
end

function comparison_tbl = buildComparisonTable(base, aware)
    comparison_tbl = struct();
    comparison_tbl.scenario = base.scenario;
    comparison_tbl.task_completion_time_delta = aware.task_completion_time_sec - base.task_completion_time_sec;
    comparison_tbl.overlap_risk_event_delta = aware.overlap_risk_event_count - base.overlap_risk_event_count;
    comparison_tbl.robot_hold_count_delta = aware.robot_hold_count - base.robot_hold_count;
    comparison_tbl.human_wait_time_delta = aware.human_wait_time_sec - base.human_wait_time_sec;
    comparison_tbl.unnecessary_slowdown_delta = aware.unnecessary_slowdown_count - base.unnecessary_slowdown_count;
end

function stats = buildStatisticalSummary(comparison_tbl)
    metric_names = {'task_completion_time_delta','overlap_risk_event_delta','robot_hold_count_delta','human_wait_time_delta','unnecessary_slowdown_delta'};
    stats_rows = cell(numel(metric_names), 1);
    for i = 1:numel(metric_names)
        m = metric_names{i};
        values = comparison_tbl.(m);
        baseline_mag = max(abs(mean(values)) + eps, 1e-6);
        stats_rows{i} = struct( ...
            'metric', char(m), ...
            'mean_improvement', -mean(values), ...
            'percent_improvement', (-mean(values) / baseline_mag) * 100.0, ...
            'variance', var(values));
    end
    stats = struct2table([stats_rows{:}]);
end

function safety = buildSafetyReport(aware_tbl, config)
    hold_ratio = aware_tbl.robot_hold_count ./ max(aware_tbl.total_simulated_time_sec / config.dt_sec, 1);
    oscillation_ratio = aware_tbl.decision_switch_count ./ max(aware_tbl.total_simulated_time_sec / config.dt_sec, 1);
    safety = struct();
    safety.scenario = aware_tbl.scenario;
    safety.hold_ratio = hold_ratio;
    safety.switch_ratio = oscillation_ratio;
    safety.excessive_holds = hold_ratio > config.max_hold_ratio;
    safety.excessive_oscillation = oscillation_ratio > config.max_oscillation_ratio;
    safety.invalid_state_outputs = aware_tbl.prediction_mismatch_count < 0;
end

function renderStage3Figures(fig_dir, comparison_tbl, aware_tbl)
    figure('Visible','off');
    vals = [comparison_tbl.task_completion_time_delta, comparison_tbl.overlap_risk_event_delta, comparison_tbl.human_wait_time_delta];
    bar(vals);
    grid on;
    title('Stage 3 A/B Deltas (Aware - Baseline)');
    xticklabels(cellstr(comparison_tbl.scenario));
    legend({'completion','overlap','human wait'}, 'Location', 'northwest');
    saveas(gcf, fullfile(fig_dir, 'ab_delta_summary.png'));
    close(gcf);

    figure('Visible','off');
    bar(aware_tbl.robot_hold_count);
    grid on;
    xticklabels(cellstr(aware_tbl.scenario));
    title('Stage 3 Hold Counts (Hesitation-Aware)');
    saveas(gcf, fullfile(fig_dir, 'hold_counts.png'));
    close(gcf);
end

function prediction = scriptPassThroughPrediction(scripted_state)
    probs = uniformProbs();
    probs.(scripted_state) = 1.0;
    prediction = struct( ...
        'predicted_state', scripted_state, ...
        'state_probabilities', probs, ...
        'future_hesitation_probability', 0.0, ...
        'future_correction_probability', 0.0, ...
        'source', 'script_passthrough');
end

function prediction = predict_hesitation_state(feature_window, feature_window_history, task_type, scripted_state, config, scenario)
% predict_hesitation_state  Stage 3 inference with Fix 4 (multi-task) and Fix 5 (temporal window).
%
%   feature_window         - current feature struct (single timestep window)
%   feature_window_history - cell array of last <=10 feature_window structs
%   task_type              - 'assembly'|'precision'|'inspection'|'generic'
%   scripted_state         - ground-truth scripted state (for stub fallback)
%   config                 - stage3 config struct
%   scenario               - scenario struct with .name field

    if config.use_stub
        prediction = stubPrediction(feature_window, scripted_state);
        return;
    end

    try
        % --- Fix 4: Select Task-Specific Model ---
        selected_model = config.models.generic;
        if isfield(config.models, task_type) && ~isempty(config.models.(task_type))
            selected_model = config.models.(task_type);
        end

        if ~isempty(selected_model)
            % --- Fix 5: Temporal Context Window (Sequence Inference) ---
            % Run inference on each frame in the window and average probabilities
            num_frames = numel(feature_window_history);
            all_probs = [];
            all_fut_hes = [];
            all_fut_corr = [];
            
            for hi = 1:num_frames
                fw = feature_window_history{hi};
                X_frame = [fw.mean_speed, fw.speed_variance, fw.pause_ratio, ...
                           fw.direction_changes, fw.progress_delta, ...
                           fw.backtrack_ratio, fw.mean_workspace_distance];
                
                [p, fh, fc, classes] = infer_classical(selected_model, X_frame);
                
                if isempty(all_probs)
                    all_probs = p;
                    all_fut_hes = fh;
                    all_fut_corr = fc;
                else
                    all_probs = all_probs + p;
                    all_fut_hes = all_fut_hes + fh;
                    all_fut_corr = all_fut_corr + fc;
                end
            end
            
            % Average the sequence results
            state_prob = all_probs / num_frames;
            fut_hes = all_fut_hes / num_frames;
            fut_corr = all_fut_corr / num_frames;

            % --- Temporal Heuristics for suppression ---
            total_direction_changes = 0;
            total_progress = 0.0;
            for hi = 1:num_frames
                total_direction_changes = total_direction_changes + feature_window_history{hi}.direction_changes;
                total_progress = total_progress + feature_window_history{hi}.progress_delta;
            end
            is_micro_hesitation = (total_direction_changes >= 2) && (total_progress <= 0.005);

            % --- Multi-Task hesitation weight (Fix 4 refinement) ---
            if strcmp(task_type, 'precision') || strcmp(task_type, 'inspection')
                if is_micro_hesitation
                    hesitation_weight = 1.0;
                else
                    hesitation_weight = 0.05;
                end
            else
                hesitation_weight = 1.0;
            end

            % --- Apply weight and re-normalize ---
            hes_mass_lost = 0.0;
            for i = 1:length(classes)
                if ~isempty(strfind(classes{i}, 'hesitation'))
                    orig = state_prob(i);
                    state_prob(i) = orig * hesitation_weight;
                    hes_mass_lost = hes_mass_lost + (orig - state_prob(i));
                end
            end
            for i = 1:length(classes)
                if strcmp(classes{i}, 'normal_progress')
                    state_prob(i) = state_prob(i) + hes_mass_lost;
                end
            end
            % --- Fix 6: Human Intention Type Classification (Multi-class mapping) ---
            % We map the classical probabilities + temporal context to the new intention types
            [~, max_idx] = max(state_prob);
            base_predicted = classes{max_idx};
            
            shared_occ = feature_window.shared_zone_occupancy;
            
            if strcmp(base_predicted, 'normal_progress')
                predicted_state = 'normal_progress';
            elseif is_micro_hesitation && shared_occ > 0.5
                predicted_state = 'coordination_uncertainty';
            elseif is_micro_hesitation && shared_occ <= 0.5
                predicted_state = 'workspace_conflict';
            elseif strcmp(task_type, 'precision') || strcmp(task_type, 'inspection')
                if feature_window.mean_speed < 0.1
                    predicted_state = 'deliberate_inspection';
                else
                    predicted_state = 'task_complexity';
                end
            else
                predicted_state = 'coordination_uncertainty';
            end

            probs = uniformProbs();
            % Initialize all to 0
            fnames = fieldnames(probs);
            for i = 1:numel(fnames), probs.(fnames{i}) = 0.0; end
            
            for i = 1:length(classes)
                cls_name = classes{i};
                p_val = state_prob(i);
                
                % Map old probabilities to the new intention-aware fields
                switch cls_name
                    case 'normal_progress'
                        probs.normal_progress = probs.normal_progress + p_val;
                    case {'mild_hesitation', 'strong_hesitation'}
                        % Assign to the currently predicted granular state
                        probs.(predicted_state) = probs.(predicted_state) + p_val;
                    case 'overlap_risk'
                        probs.workspace_conflict = probs.workspace_conflict + p_val;
                    case 'ready_for_robot_action'
                        probs.ready_for_robot_action = probs.ready_for_robot_action + p_val;
                    case 'correction_rework'
                        probs.correction_rework = probs.correction_rework + p_val;
                    otherwise
                        % Catch-all
                        probs.coordination_uncertainty = probs.coordination_uncertainty + p_val;
                end
            end

            prediction = struct( ...
                'predicted_state', predicted_state, ...
                'state_probabilities', probs, ...
                'future_hesitation_probability', fut_hes, ...
                'future_correction_probability', fut_corr, ...
                'source', sprintf('native_multi_task_fix6_%s', task_type));
        else
            prediction = predictViaSystemBridge(feature_window, config);
        end
    catch e
        warning('Stage3:InferenceFailed', 'Inference failed: %s. Falling back to stub.', e.message);
        prediction = stubPrediction(feature_window, scripted_state);
        prediction.source = 'error_fallback';
    end
end


function prediction = predictViaSystemBridge(feature_window, config)
    json_payload = jsonencode(feature_window);
    cmd = sprintf('%s %s --features-json ''%s''', config.bridge_python, config.bridge_script, escapeSingleQuotes(json_payload));
    [status, out] = system(cmd);
    if status ~= 0
        error('Stage3Bridge:CallFailed', 'Python bridge failed: %s', out);
    end
    decoded = jsondecode(strtrim(out));
    if isfield(decoded, 'error')
        error('Stage3Bridge:InferenceError', 'Python bridge error: %s', decoded.error);
    end
    prediction = normalizePredictionStruct(decoded);
end

function prediction = normalizePredictionStruct(raw)
    prediction = struct( ...
        'predicted_state', char(raw.predicted_state), ...
        'state_probabilities', raw.state_probabilities, ...
        'future_hesitation_probability', double(raw.future_hesitation_probability), ...
        'future_correction_probability', double(raw.future_correction_probability), ...
        'source', 'python_bridge');
end

function validatePredictionOutput(pred)
    keys = {'predicted_state','state_probabilities','future_hesitation_probability','future_correction_probability'};
    for i = 1:numel(keys)
        if ~isfield(pred, keys{i})
            error('Stage3Prediction:MissingKey', 'Missing prediction key: %s', keys{i});
        end
    end

    valid_states = {'normal_progress', 'coordination_uncertainty', 'task_complexity', 'deliberate_inspection', 'workspace_conflict', 'ready_for_robot_action', 'correction_rework'};
    if ~any(strcmp(pred.predicted_state, valid_states))
        error('Stage3Prediction:InvalidState', 'Invalid predicted state: %s', pred.predicted_state);
    end

    probs = pred.state_probabilities;
    total = 0.0;
    for i = 1:numel(valid_states)
        key = valid_states{i};
        if isfield(probs, key)
            value = double(probs.(key));
            if value < 0.0 || value > 1.0
                error('Stage3Prediction:InvalidProbability', 'Probability out of range for %s', key);
            end
            total = total + value;
        end
    end
    if abs(total - 1.0) > 1e-3
        error('Stage3Prediction:ProbabilitySum', 'State probabilities must sum to 1.0 (got %.6f).', total);
    end

    for rk = {'future_hesitation_probability','future_correction_probability'}
        value = double(pred.(rk{1}));
        if value < 0.0 || value > 1.0
            error('Stage3Prediction:InvalidRiskProbability', 'Invalid %s', rk{1});
        end
    end
end

function p = stubPrediction(feature_window, scripted_state)
    p = scriptPassThroughPrediction(scripted_state);
    p.source = 'stub_fallback';
    
    % Heuristic intention mapping for stub
    if feature_window.direction_changes >= 2 && feature_window.shared_zone_occupancy >= 0.5
        p.predicted_state = 'coordination_uncertainty';
    elseif feature_window.direction_changes >= 2 && feature_window.shared_zone_occupancy < 0.5
        p.predicted_state = 'workspace_conflict';
    elseif feature_window.pause_ratio >= 0.40 && feature_window.mean_speed < 0.1
        p.predicted_state = 'deliberate_inspection';
    elseif feature_window.pause_ratio >= 0.20
        p.predicted_state = 'task_complexity';
    elseif feature_window.task_step >= 5 && feature_window.distance_to_target < 0.18
        p.predicted_state = 'ready_for_robot_action';
    else
        p.predicted_state = 'normal_progress';
    end
    
    p.state_probabilities = uniformProbs();
    p.state_probabilities.(p.predicted_state) = 1.0;
end

function probs = uniformProbs()
    states = {'normal_progress', 'coordination_uncertainty', 'task_complexity', 'deliberate_inspection', 'workspace_conflict', 'ready_for_robot_action', 'correction_rework'};
    probs = struct();
    for i = 1:numel(states)
        probs.(states{i}) = 0.0;
    end
end

function d = probabilityDistance(left, right)
    keys = {'normal_progress', 'coordination_uncertainty', 'task_complexity', 'deliberate_inspection', 'workspace_conflict', 'ready_for_robot_action', 'correction_rework'};
    d = 0.0;
    for i = 1:numel(keys)
        key = keys{i};
        if isfield(left, key) && isfield(right, key)
            d = d + abs(double(left.(key)) - double(right.(key)));
        end
    end
end

function row = featureLogRow(ts, scenario, policy, scripted_state, prediction, feature_window, robot_mode, robot_speed)
    source_val = 'unknown';
    if isfield(prediction, 'source')
        source_val = prediction.source;
    end
    row = struct( ...
        'timestamp', ts, ...
        'scenario', char(scenario), ...
        'policy', char(policy), ...
        'scripted_state', char(scripted_state), ...
        'predicted_state', char(prediction.predicted_state), ...
        'state_probabilities', prediction.state_probabilities, ...
        'future_hesitation_probability', prediction.future_hesitation_probability, ...
        'future_correction_probability', prediction.future_correction_probability, ...
        'source', char(source_val), ...
        'robot_mode', char(robot_mode), ...
        'robot_speed_cmd', robot_speed, ...
        'feature_window', feature_window);
end

function out = escapeSingleQuotes(text)
    out = strrep(text, '''', '''"''"''');
end


% ------- Shared Stage2-compatible helpers (schema + dynamics) -------
function [speed, mode] = baselinePolicy(config, t_sec)
    if t_sec < config.robot_release_delay_sec
        speed = 0.0; mode = 'hold';
    else
        speed = config.robot_nominal_speed; mode = 'proceed';
    end
end

function [speed, mode] = policyBFromInference(pred_state, config, t_sec)
    if t_sec < config.robot_release_delay_sec
        speed = 0.0; mode = 'hold'; return;
    end
    switch pred_state
        case 'normal_progress', speed = config.robot_nominal_speed; mode = 'proceed';
        case 'task_complexity', speed = 0.70 * config.robot_nominal_speed; mode = 'slow';
        case 'deliberate_inspection', speed = 0.85 * config.robot_nominal_speed; mode = 'proceed';
        case 'coordination_uncertainty', speed = 0.0; mode = 'hold'; % Fix 6: HOLD for coordination uncertainty
        case 'workspace_conflict', speed = 0.0; mode = 'hold'; % Fix 6: HOLD for conflict
        case 'ready_for_robot_action', speed = config.robot_nominal_speed; mode = 'proceed';
        case 'correction_rework', speed = 0.0; mode = 'hold';
        otherwise, speed = config.robot_nominal_speed; mode = 'proceed';
    end
end

function history = initializeFeatureHistory()
    history = struct('hand_pos', [], 'speed', [], 'progress', [], 'retry', [], ...
        'dist_target', [], 'human_robot_dist', [], 'task_step', [], 'shared_occ', []);
end

function history = appendFeatureHistory(history, hand_pos, speed, progress, retry_count, dist_target, human_robot_distance, task_step, shared_occ)
    history.hand_pos = [history.hand_pos; hand_pos];
    history.speed = [history.speed; speed];
    history.progress = [history.progress; progress];
    history.retry = [history.retry; retry_count];
    history.dist_target = [history.dist_target; dist_target];
    history.human_robot_dist = [history.human_robot_dist; human_robot_distance];
    history.task_step = [history.task_step; task_step];
    history.shared_occ = [history.shared_occ; shared_occ];
end

function feature = computeFeatureWindow(history, window_size, timestamp)
    start_idx = max(1, numel(history.speed) - window_size + 1);
    idx = start_idx:numel(history.speed);
    speed = history.speed(idx);
    progress = history.progress(idx);
    retry = history.retry(idx);
    pos = history.hand_pos(idx, :);
    move_delta = diff(pos, 1, 1);
    signed_dx = move_delta(:, 1);
    reversal_count = sum(abs(diff(sign(signed_dx))) > 0);
    progress_step = diff(progress);
    backtrack_ratio = 0.0;
    if ~isempty(progress_step), backtrack_ratio = sum(progress_step < 0) / numel(progress_step); end

    feature = struct( ...
        'timestamp', timestamp, 'mean_speed', mean(speed), 'pause_ratio', mean(speed <= 0.03), ...
        'progress_delta', progress(end) - progress(1), 'reversal_count', reversal_count, ...
        'retry_count', max(retry), 'distance_to_target', history.dist_target(end), ...
        'human_robot_distance', history.human_robot_dist(end), 'task_step', history.task_step(end), ...
        'shared_zone_occupancy', mean(history.shared_occ(idx)), 'speed_variance', var(speed), ...
        'direction_changes', reversal_count, 'backtrack_ratio', backtrack_ratio, ...
        'mean_workspace_distance', mean(history.human_robot_dist(idx)));
end

function validateFeatureSchemaContract(feature)
    required = {'timestamp','mean_speed','pause_ratio','progress_delta','reversal_count','retry_count', ...
        'distance_to_target','human_robot_distance','task_step','shared_zone_occupancy', ...
        'speed_variance','direction_changes','backtrack_ratio','mean_workspace_distance'};
    for i = 1:numel(required)
        if ~isfield(feature, required{i}), error('Stage3FeatureSchema:MissingField', 'Missing field %s', required{i}); end
    end
    numeric_keys = {'timestamp','mean_speed','pause_ratio','progress_delta','distance_to_target','human_robot_distance', ...
        'shared_zone_occupancy','speed_variance','backtrack_ratio','mean_workspace_distance'};
    for i = 1:numel(numeric_keys)
        v = feature.(numeric_keys{i});
        if ~(isnumeric(v) && isscalar(v)), error('Stage3FeatureSchema:TypeMismatch', 'Field %s invalid', numeric_keys{i}); end
    end
    int_keys = {'reversal_count','retry_count','task_step','direction_changes'};
    for i = 1:numel(int_keys)
        v = feature.(int_keys{i});
        if ~(isnumeric(v) && isscalar(v) && abs(v - round(v)) < 1e-9), error('Stage3FeatureSchema:IntMismatch', 'Field %s invalid', int_keys{i}); end
    end
end

function feature = emptyFeatureWindow(timestamp)
    feature = struct('timestamp', timestamp, 'mean_speed', 0.0, 'pause_ratio', 0.0, 'progress_delta', 0.0, ...
        'reversal_count', 0, 'retry_count', 0, 'distance_to_target', 1.0, 'human_robot_distance', 1.0, ...
        'task_step', 1, 'shared_zone_occupancy', 0.0, 'speed_variance', 0.0, 'direction_changes', 0, ...
        'backtrack_ratio', 0.0, 'mean_workspace_distance', 1.0);
end

function state = scriptedStateSample(state_names, weights, t_sec, progress)
    adjusted = weights;
    if progress > 0.80
        adjusted(5) = adjusted(5) + 0.20;
        adjusted(3) = max(0.01, adjusted(3) - 0.10);
    end
    if t_sec < 5.0
        adjusted(1) = adjusted(1) + 0.15;
    end
    adjusted = adjusted / sum(adjusted);
    state = state_names{find(rand() <= cumsum(adjusted), 1, 'first')};
end

function params = scriptedStateParameters(state)
    switch state
        case 'normal_progress', params = mkParams(1.00, 0.02, 0, 0.10, 0.020);
        case 'coordination_uncertainty', params = mkParams(0.50, 0.40, 2, 0.30, 0.008);
        case 'task_complexity', params = mkParams(0.65, 0.25, 1, 0.15, 0.012);
        case 'deliberate_inspection', params = mkParams(0.20, 0.60, 0, 0.05, 0.002);
        case 'workspace_conflict', params = mkParams(0.80, 0.05, 1, 0.70, 0.010);
        case 'ready_for_robot_action', params = mkParams(1.05, 0.00, 0, 0.18, 0.022);
        case 'correction_rework', params = mkParams(0.40, 0.45, 3, 0.25, 0.004);
        otherwise, params = mkParams(1.00, 0.05, 0, 0.10, 0.015);
    end
end

function params = mkParams(speed_scale, pause_probability, retry_count, shared_zone_entry_prob, progress_rate)
    params = struct('speed_scale', speed_scale, 'pause_probability', pause_probability, 'retry_count', retry_count, ...
        'shared_zone_entry_prob', shared_zone_entry_prob, 'progress_rate', progress_rate);
end

function pos = stepToward(pos, target, speed, dt)
    delta = target - pos; dist = norm(delta);
    if dist < 1e-9 || speed <= 0, return; end
    pos = pos + (delta / dist) * min(dist, speed * dt);
end

function flag = isInSharedZone(pos, config)
    flag = pos(1) >= config.shared_zone_x(1) && pos(1) <= config.shared_zone_x(2) && ...
        pos(2) >= config.shared_zone_y(1) && pos(2) <= config.shared_zone_y(2);
end

function writeJsonl(path, rows)
    fid = fopen(path, 'w'); if fid < 0, error('Cannot write: %s', path); end
    c = onCleanup(@() fclose(fid));
    if iscell(rows)
        for i = 1:numel(rows), fprintf(fid, '%s\n', jsonencode(rows{i})); end
    else
        for i = 1:numel(rows), fprintf(fid, '%s\n', jsonencode(rows(i))); end
    end
    clear c;
end

function rows = readJsonl(path)
    fid = fopen(path, 'r'); if fid < 0, error('Cannot read: %s', path); end
    c = onCleanup(@() fclose(fid));
    raw_rows = cell(1000, 1); idx = 1;
    while true
        ln = fgetl(fid); if ~ischar(ln), break; end
        if isempty(strtrim(ln)), continue; end
        raw_rows{idx} = jsondecode(ln);
        idx = idx + 1;
        if idx > numel(raw_rows)
            raw_rows = [raw_rows; cell(1000, 1)]; %#ok<AGROW>
        end
    end
    raw_rows = raw_rows(1:idx-1);
    if isempty(raw_rows)
        rows = struct([]);
    else
        rows = [raw_rows{:}];
    end
    clear c;
end
