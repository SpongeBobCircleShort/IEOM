classdef HesitationModelCLI
    % HesitationModelCLI: Wrapper for calling Python hesitation model from MATLAB
    % 
    % Usage:
    %   cli = HesitationModelCLI('/path/to/ieom_model');
    %   prediction = cli.predict_single(features_struct);
    
    properties
        model_root_dir
        python_cmd = 'python3'
    end
    
    properties (Constant)
        MODULE_PATH = 'src'
        POLICY_MAPPING = struct(...
            'normal_progress', 1.0, ...
            'mild_hesitation', 0.8, ...
            'strong_hesitation', 0.5, ...
            'correction_rework', 0.0, ...
            'ready_for_robot_action', 1.0, ...
            'overlap_risk', 0.3 ...
        )
    end
    
    methods
        function obj = HesitationModelCLI(model_root_dir)
            % Initialize CLI wrapper
            obj.model_root_dir = model_root_dir;
            
            % Verify model directory exists
            if ~isdir(model_root_dir)
                error(['Model directory not found: ', model_root_dir]);
            end
        end
        
        function prediction = predict_single(obj, features)
            % predict_single: Get prediction for a single feature vector
            %
            % Args:
            %   features: struct with 7 fields:
            %     - mean_hand_speed (float)
            %     - pause_ratio (float)
            %     - progress_delta (float)
            %     - reversal_count (int32)
            %     - retry_count (int32)
            %     - task_step_id (int32)
            %     - human_robot_distance (float)
            %
            % Returns:
            %   prediction: struct with fields:
            %     - state (string): One of 6 states
            %     - state_probabilities (struct): Probabilities for each state
            %     - future_hesitation_prob (float): [0-1]
            %     - future_correction_prob (float): [0-1]
            %     - confidence (float): Max state probability
            %     - window_size_frames (int)
            %     - frame_rate_hz (int)
            
            % Build CLI command
            cmd = sprintf(...
                ['PYTHONPATH=%s %s -m hesitation.inference.cli predict ', ...
                '--mean-hand-speed %.6f --speed-variance %.6f --pause-ratio %.6f ', ...
                '--progress-delta %.6f --direction-changes %d --reversal-count %d ', ...
                '--backtrack-ratio %.6f --retry-count %d --task-step-id %d ', ...
                '--human-robot-distance %.6f --mean-workspace-distance %.6f'], ...
                fullfile(obj.model_root_dir, obj.MODULE_PATH), ...
                obj.python_cmd, ...
                features.mean_hand_speed, ...
                getFeatureValue(features, 'speed_variance', 0.0), ...
                features.pause_ratio, ...
                features.progress_delta, ...
                int32(getFeatureValue(features, 'direction_changes', features.reversal_count)), ...
                features.reversal_count, ...
                getFeatureValue(features, 'backtrack_ratio', min(double(features.retry_count), 1.0)), ...
                features.retry_count, ...
                features.task_step_id, ...
                features.human_robot_distance, ...
                getFeatureValue(features, 'mean_workspace_distance', features.human_robot_distance) ...
            );
            
            % Execute command from model directory
            original_dir = pwd;
            cd(obj.model_root_dir);
            
            try
                [status, result] = system(cmd);
                
                if status ~= 0
                    error(['CLI execution failed: ', result]);
                end
                
                % Parse JSON output (last line, skip warnings)
                lines = strsplit(strtrim(result), newline);
                json_output = lines{end};
                
                % Decode JSON
                prediction = jsondecode(json_output);
                
                % Add robot action speed factor
                state = prediction.state;
                if isfield(obj.POLICY_MAPPING, state)
                    prediction.robot_speed_factor = obj.POLICY_MAPPING.(state);
                else
                    warning(['Unknown state: ', state]);
                    prediction.robot_speed_factor = 0.5;  % Default to safe speed
                end
                
            catch ME
                cd(original_dir);
                rethrow(ME);
            end
            
            cd(original_dir);
        end
        
        function [state, speed_factor] = get_robot_action(obj, state_str)
            % get_robot_action: Map predicted state to robot action
            %
            % Args:
            %   state_str: Predicted hesitation state (string)
            %
            % Returns:
            %   state: Confirmed state string
            %   speed_factor: Robot speed multiplier [0.0, 1.0]
            
            if isfield(obj.POLICY_MAPPING, state_str)
                speed_factor = obj.POLICY_MAPPING.(state_str);
                state = state_str;
            else
                warning(['Unknown state: ', state_str, '; defaulting to 0.5× speed']);
                speed_factor = 0.5;
                state = state_str;
            end
        end
    end
end

function value = getFeatureValue(features, field_name, default_value)
    if isfield(features, field_name)
        value = features.(field_name);
    else
        value = default_value;
    end
end
