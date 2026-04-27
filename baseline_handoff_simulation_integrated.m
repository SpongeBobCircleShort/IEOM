function baseline_handoff_simulation_integrated()
    % baseline_handoff_simulation_integrated: MATLAB simulator with hesitation model integration
    %
    % This script demonstrates the complete integration:
    % 1. Extract features from simulated kinematics
    % 2. Call Python hesitation model via CLI
    % 3. Apply policy mapping to robot actions
    % 4. Log predictions and actions to JSONL
    
    fprintf('=== IEOM Simulator with Hesitation Model Integration ===\n\n');
    
    % Configuration
    ieom_model_root = '/Users/adijain/ENGINEERING/IEOM/ieom_model';
    experiment_output_dir = '/tmp/ieom_experiments';
    
    if ~isdir(experiment_output_dir)
        mkdir(experiment_output_dir);
    end
    
    % Initialize components
    feature_extractor = FeatureExtractor();
    model_cli = HesitationModelCLI(ieom_model_root);
    
    % Run scenarios
    scenarios = {
        struct('name', 'scenario_a', 'title', 'Scenario A: Normal Progress'), ...
        struct('name', 'scenario_b', 'title', 'Scenario B: Mild Hesitation'), ...
        struct('name', 'scenario_c', 'title', 'Scenario C: Strong Hesitation'), ...
        struct('name', 'scenario_d', 'title', 'Scenario D: Correction/Rework') ...
    };
    
    for s = 1:length(scenarios)
        fprintf('\n%s\n', scenarios{s}.title);
        run_scenario(scenarios{s}, feature_extractor, model_cli, experiment_output_dir, ieom_model_root);
    end
    
    fprintf('\n✓ All scenarios complete. Logs saved to: %s\n', experiment_output_dir);
end

function run_scenario(scenario, feature_extractor, model_cli, output_dir, model_root)
    % run_scenario: Execute one scenario with logging
    
    % Generate trial ID
    trial_id = [scenario.name, '_', datestr(now, 'yyyymmdd_HHMMSS')];
    logger = ExperimentLogger(trial_id, output_dir);
    
    % Scenario parameters
    switch scenario.name
        case 'scenario_a'
            % Normal progress: smooth motion
            num_frames = 100;
            hand_trajectory = generate_normal_trajectory(num_frames);
            
        case 'scenario_b'
            % Mild hesitation: brief pause
            num_frames = 100;
            hand_trajectory = generate_hesitation_trajectory(num_frames, 0.2);
            
        case 'scenario_c'
            % Strong hesitation: prolonged pause
            num_frames = 120;
            hand_trajectory = generate_hesitation_trajectory(num_frames, 0.6);
            
        case 'scenario_d'
            % Correction/rework: reversals and restarts
            num_frames = 150;
            hand_trajectory = generate_rework_trajectory(num_frames);
    end
    
    % Initialize metrics
    collision_count = 0;
    min_distance = inf;
    proximity_warnings = 0;
    task_restart_flags = false(1, num_frames);
    
    % Simulate robot TCP at fixed position (for demo)
    robot_pos = [0.5, 0.5, 0.5];
    
    % Run trial frame-by-frame
    tic;
    for frame_idx = 1:num_frames
        timestamp_sec = toc;
        
        % Get current hand position
        hand_pos = hand_trajectory(frame_idx, :);
        
        % Compute task progress
        progress = frame_idx / num_frames;
        task_step = min(floor((frame_idx / num_frames) * 20), 20);
        
        % Detect task restart (demo: restart at specific frame for scenario D)
        task_restart = false;
        if strcmp(scenario.name, 'scenario_d') && (frame_idx == 60 || frame_idx == 100)
            task_restart = true;
        end
        task_restart_flags(frame_idx) = task_restart;
        
        % 1. EXTRACT FEATURES
        features = feature_extractor.extract_features(...
            hand_pos, robot_pos, progress, task_step, task_restart ...
        );
        
        % 2. GET PREDICTION
        try
            prediction = model_cli.predict_single(features);
            
            % 3. LOG PREDICTION
            logger.log_prediction(timestamp_sec, frame_idx, features, prediction);
            
            % 4. GET ROBOT ACTION
            state = prediction.state;
            speed_factor = prediction.robot_speed_factor;
            
            % Map state to robot delay
            delay_map = struct(...
                'normal_progress', 0, ...
                'mild_hesitation', 100, ...
                'strong_hesitation', 200, ...
                'correction_rework', 500, ...
                'ready_for_robot_action', 0, ...
                'overlap_risk', inf ...
            );
            
            if isfield(delay_map, state)
                delay_ms = delay_map.(state);
            else
                delay_ms = 0;
            end
            
            % Safety metrics for this frame
            hand_robot_dist = norm(hand_pos - robot_pos);
            if hand_robot_dist < min_distance
                min_distance = hand_robot_dist;
            end
            
            if hand_robot_dist < 0.2
                proximity_warnings = proximity_warnings + 1;
            end
            
            % 5. LOG POLICY ACTION
            robot_action = struct(...
                'speed_factor', speed_factor, ...
                'delay_ms', delay_ms, ...
                'action_name', state ...
            );
            
            safety_metrics = struct(...
                'hand_robot_distance', hand_robot_dist, ...
                'collision_detected', false, ...
                'proximity_warning', hand_robot_dist < 0.2 ...
            );
            
            logger.log_policy_action(timestamp_sec, frame_idx, state, robot_action, safety_metrics);
            
        catch ME
            warning(['Frame %d prediction failed: %s'], frame_idx, ME.message);
        end
        
        % Print progress every 20 frames
        if mod(frame_idx, 20) == 0
            fprintf('  Frame %d/%d | State: %s | Speed: %.1f%% | Hand-Robot: %.2f m\n', ...
                frame_idx, num_frames, state, speed_factor * 100, hand_robot_dist);
        end
    end
    
    trial_time = toc;
    
    % Save metrics
    logger.save_trial_metadata(scenario.name, 'v1.0_dummy', struct('version', 'R2024a'), 12345);
    logger.save_safety_metrics(collision_count, min_distance, proximity_warnings);
    logger.save_efficiency_metrics(trial_time, 0, 0);
    logger.close();
    
    fprintf('  ✓ Trial complete: %s | Time: %.1f s | Min distance: %.2f m | Warnings: %d\n', ...
        trial_id, trial_time, min_distance, proximity_warnings);
end

function trajectory = generate_normal_trajectory(num_frames)
    % Generate smooth forward motion
    trajectory = zeros(num_frames, 3);
    for i = 1:num_frames
        t = i / num_frames;
        trajectory(i, 1) = 0.5 + t * 0.4;  % x: 0.5 -> 0.9 (moving forward)
        trajectory(i, 2) = 0.5 + 0.05 * sin(2 * pi * t);  % y: small oscillation
        trajectory(i, 3) = 0.5;  % z: constant
    end
end

function trajectory = generate_hesitation_trajectory(num_frames, hesitation_factor)
    % Generate motion with pauses and reversals
    trajectory = zeros(num_frames, 3);
    for i = 1:num_frames
        t = i / num_frames;
        
        % Add hesitation: reduce speed and add reversals
        if mod(floor(t * 10), 3) == 0
            % Pause or slight reversal
            trajectory(i, 1) = 0.5 + (t - hesitation_factor) * 0.4;
        else
            trajectory(i, 1) = 0.5 + t * 0.4;
        end
        
        trajectory(i, 2) = 0.5 + 0.1 * sin(2 * pi * t);
        trajectory(i, 3) = 0.5;
    end
end

function trajectory = generate_rework_trajectory(num_frames)
    % Generate trajectory with multiple reversals and restarts
    trajectory = zeros(num_frames, 3);
    restart_points = [1, 60, 100];  % Frame indices of task restarts
    
    for i = 1:num_frames
        % Find which restart phase we're in
        phase = 1;
        for j = length(restart_points):-1:1
            if i >= restart_points(j)
                phase = j;
                break;
            end
        end
        
        % Position within phase
        phase_start = restart_points(phase);
        phase_duration = (i - phase_start + 1);
        phase_progress = phase_duration / 60;  % Assume 60-frame phases
        
        % Forward progress with reversals
        x = 0.5 + phase_progress * 0.3;
        
        % Add reversals
        if mod(phase_duration, 15) < 3
            x = x - 0.05;  % Slight reversal
        end
        
        trajectory(i, 1) = min(max(x, 0.3), 0.8);  % Clamp
        trajectory(i, 2) = 0.5 + 0.15 * sin(2 * pi * phase_progress);
        trajectory(i, 3) = 0.5;
    end
end
