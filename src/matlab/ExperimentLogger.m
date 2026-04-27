classdef ExperimentLogger
    % ExperimentLogger: Log predictions and robot actions to JSONL format
    % 
    % Usage:
    %   logger = ExperimentLogger(trial_id, output_dir);
    %   logger.log_prediction(timestamp, frame_idx, features, prediction);
    %   logger.log_policy_action(timestamp, frame_idx, state, action, metrics);
    %   logger.save_trial_metadata(config);
    %   logger.close();
    
    properties
        trial_id
        output_dir
        trial_dir
        prediction_log_fid
        policy_log_fid
        trial_start_time
    end
    
    methods
        function obj = ExperimentLogger(trial_id, output_dir)
            % Initialize logger for a trial
            obj.trial_id = trial_id;
            obj.output_dir = output_dir;
            obj.trial_start_time = datetime('now');
            
            % Create trial directory
            obj.trial_dir = fullfile(output_dir, trial_id);
            if ~isdir(obj.trial_dir)
                mkdir(obj.trial_dir);
            end
            
            % Open log files
            obj.prediction_log_fid = fopen(fullfile(obj.trial_dir, 'predictions.jsonl'), 'w');
            obj.policy_log_fid = fopen(fullfile(obj.trial_dir, 'policy_actions.jsonl'), 'w');
            
            if obj.prediction_log_fid < 0 || obj.policy_log_fid < 0
                error(['Failed to open log files in: ', obj.trial_dir]);
            end
        end
        
        function log_prediction(obj, timestamp_sec, frame_idx, features, prediction)
            % log_prediction: Log model prediction for a frame
            %
            % Args:
            %   timestamp_sec: Time in seconds since trial start
            %   frame_idx: Frame number in trial
            %   features: Feature struct (7 features)
            %   prediction: Prediction struct from model
            
            entry = struct(...
                'trial_id', obj.trial_id, ...
                'timestamp_sec', timestamp_sec, ...
                'frame_idx', frame_idx, ...
                'input_features', features, ...
                'model_output', prediction ...
            );
            
            json_str = jsonencode(entry);
            fprintf(obj.prediction_log_fid, '%s\n', json_str);
        end
        
        function log_policy_action(obj, timestamp_sec, frame_idx, state, robot_action, safety_metrics)
            % log_policy_action: Log robot action taken based on prediction
            %
            % Args:
            %   timestamp_sec: Time in seconds since trial start
            %   frame_idx: Frame number
            %   state: Predicted hesitation state (string)
            %   robot_action: struct with fields:
            %     - speed_factor (float)
            %     - delay_ms (float)
            %     - action_name (string)
            %   safety_metrics: struct with safety data
            
            entry = struct(...
                'trial_id', obj.trial_id, ...
                'timestamp_sec', timestamp_sec, ...
                'frame_idx', frame_idx, ...
                'predicted_state', state, ...
                'robot_action', robot_action, ...
                'safety_metrics', safety_metrics ...
            );
            
            json_str = jsonencode(entry);
            fprintf(obj.policy_log_fid, '%s\n', json_str);
        end
        
        function save_trial_metadata(obj, scenario, model_version, simulator_config, random_seed)
            % save_trial_metadata: Save trial configuration and metadata
            %
            % Args:
            %   scenario: Scenario name (string)
            %   model_version: Model checkpoint identifier
            %   simulator_config: Simulator version and settings (struct)
            %   random_seed: Random seed for reproducibility
            
            metadata = struct(...
                'trial_id', obj.trial_id, ...
                'timestamp_start', obj.trial_start_time, ...
                'timestamp_end', datetime('now'), ...
                'scenario', scenario, ...
                'model_version', model_version, ...
                'simulator_config', simulator_config, ...
                'random_seed', random_seed ...
            );
            
            metadata_file = fullfile(obj.trial_dir, 'trial_metadata.json');
            json_str = jsonencode(metadata);
            fid = fopen(metadata_file, 'w');
            fprintf(fid, '%s\n', json_str);
            fclose(fid);
        end
        
        function save_safety_metrics(obj, collision_count, min_distance, proximity_warnings)
            % save_safety_metrics: Save trial safety metrics
            
            metrics = struct(...
                'trial_id', obj.trial_id, ...
                'collision_count', collision_count, ...
                'min_hand_robot_distance', min_distance, ...
                'proximity_warnings', proximity_warnings ...
            );
            
            file = fullfile(obj.trial_dir, 'safety_metrics.json');
            json_str = jsonencode(metrics);
            fid = fopen(file, 'w');
            fprintf(fid, '%s\n', json_str);
            fclose(fid);
        end
        
        function save_efficiency_metrics(obj, completion_time_sec, total_pause_time_sec, reversal_time_sec)
            % save_efficiency_metrics: Save trial efficiency metrics
            
            metrics = struct(...
                'trial_id', obj.trial_id, ...
                'task_completion_time_sec', completion_time_sec, ...
                'total_pause_time_sec', total_pause_time_sec, ...
                'total_reversal_time_sec', reversal_time_sec, ...
                'progress_per_second', 1.0 / max(completion_time_sec, 1e-6) ...
            );
            
            file = fullfile(obj.trial_dir, 'efficiency_metrics.json');
            json_str = jsonencode(metrics);
            fid = fopen(file, 'w');
            fprintf(fid, '%s\n', json_str);
            fclose(fid);
        end
        
        function close(obj)
            % close: Flush and close all log files
            if obj.prediction_log_fid > 0
                fclose(obj.prediction_log_fid);
            end
            if obj.policy_log_fid > 0
                fclose(obj.policy_log_fid);
            end
        end
    end
end
