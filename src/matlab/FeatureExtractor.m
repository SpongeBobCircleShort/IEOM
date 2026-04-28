classdef FeatureExtractor
    % FeatureExtractor: Extract 7 features for hesitation model from simulator data
    % 
    % Features:
    %   - mean_hand_speed: Average hand velocity (m/s) [0, 2.0]
    %   - pause_ratio: Fraction of frames with near-zero velocity [0, 1.0]
    %   - progress_delta: Fractional progress toward goal [0, 1.0]
    %   - reversal_count: Number of direction reversals [0, 10]
    %   - retry_count: Cumulative task restarts [0, 5]
    %   - task_step_id: Current assembly step (0-indexed) [0, 20]
    %   - human_robot_distance: Min hand-to-TCP distance (m) [0, 2.0]
    
    properties (Constant)
        WINDOW_SIZE = 20              % frames
        FRAME_RATE = 10               % Hz
        VELOCITY_THRESHOLD = 0.05     % m/s - below this = pause
        MAX_REVERSAL_COUNT = 10
        MAX_RETRY_COUNT = 5
    end
    
    properties
        hand_position_history         % [idx, x, y, z] buffer
        hand_velocity_history         % [idx, vx, vy, vz] buffer
        window_idx = 0                % current position in circular buffer
        retry_count = 0               % cumulative restarts
    end
    
    methods
        function obj = FeatureExtractor()
            % Initialize buffers
            obj.hand_position_history = zeros(FeatureExtractor.WINDOW_SIZE, 4);
            obj.hand_velocity_history = zeros(FeatureExtractor.WINDOW_SIZE, 4);
        end
        
        function features = extract_features(obj, ...
                hand_pos_xyz, robot_pos_xyz, progress, task_step, task_restart)
            % extract_features: Compute all 7 features for current frame
            %
            % Args:
            %   hand_pos_xyz: [x, y, z] hand position (m)
            %   robot_pos_xyz: [x, y, z] robot TCP position (m)
            %   progress: scalar [0, 1] - fractional task progress
            %   task_step: integer [0, 20] - current step index
            %   task_restart: boolean - true if restart detected this frame
            %
            % Returns:
            %   features: struct with all 7 features
            
            % Update circular buffer index
            obj.window_idx = mod(obj.window_idx, FeatureExtractor.WINDOW_SIZE) + 1;
            
            % 1. Compute hand velocity
            if obj.window_idx == 1
                % First frame: use zero velocity
                hand_velocity = [0, 0, 0];
            else
                prev_idx = obj.window_idx - 1;
                if prev_idx == 0
                    prev_idx = FeatureExtractor.WINDOW_SIZE;
                end
                
                dt = 1.0 / FeatureExtractor.FRAME_RATE;
                prev_pos = obj.hand_position_history(prev_idx, 2:4);
                hand_velocity = (hand_pos_xyz - prev_pos) / dt;
            end
            
            % Store in history
            obj.hand_position_history(obj.window_idx, :) = [obj.window_idx, hand_pos_xyz];
            obj.hand_velocity_history(obj.window_idx, :) = [obj.window_idx, hand_velocity];
            
            % 2. mean_hand_speed: Average speed over window
            speeds = vecnorm(obj.hand_velocity_history(:, 2:4), 2, 2);
            mean_hand_speed = mean(speeds);
            mean_hand_speed = min(mean_hand_speed, 2.0);  % Clamp to [0, 2.0]
            
            % 3. pause_ratio: Fraction of frames with near-zero velocity
            pause_mask = speeds < FeatureExtractor.VELOCITY_THRESHOLD;
            pause_ratio = sum(pause_mask) / FeatureExtractor.WINDOW_SIZE;
            
            % 4. reversal_count: Count direction reversals in window
            reversal_count = obj.count_reversals();
            reversal_count = min(reversal_count, FeatureExtractor.MAX_REVERSAL_COUNT);
            
            % 5. retry_count: Cumulative task restarts
            if task_restart
                obj.retry_count = obj.retry_count + 1;
            end
            retry_count = min(obj.retry_count, FeatureExtractor.MAX_RETRY_COUNT);
            
            % 6. progress_delta: Fractional task progress
            progress_delta = progress;
            
            % 7. task_step_id: Current task step
            task_step_id = task_step;
            
            % 8. human_robot_distance: Min hand-to-TCP distance
            hand_robot_dist = norm(hand_pos_xyz - robot_pos_xyz);
            hand_robot_dist = min(hand_robot_dist, 2.0);  % Clamp to [0, 2.0]
            
            % Return struct with all features
            features = struct(...
                'mean_hand_speed', mean_hand_speed, ...
                'pause_ratio', pause_ratio, ...
                'progress_delta', progress_delta, ...
                'reversal_count', int32(reversal_count), ...
                'retry_count', int32(retry_count), ...
                'task_step_id', int32(task_step_id), ...
                'human_robot_distance', hand_robot_dist ...
            );
        end
        
        function reversal_count = count_reversals(obj)
            % count_reversals: Count direction reversals in window
            % Only count reversals where both samples exceed velocity threshold
            
            reversal_count = 0;
            velocities = obj.hand_velocity_history(:, 2:4);
            speeds = vecnorm(velocities, 2, 2);
            
            for i = 1:(FeatureExtractor.WINDOW_SIZE - 1)
                curr_speed = speeds(i);
                next_speed = speeds(i + 1);
                
                % Only count if both above threshold
                if curr_speed > FeatureExtractor.VELOCITY_THRESHOLD && ...
                   next_speed > FeatureExtractor.VELOCITY_THRESHOLD
                    
                    % Check if direction changed
                    curr_vel = velocities(i, :);
                    next_vel = velocities(i + 1, :);
                    
                    % Check for sign change in primary direction
                    dot_prod = dot(curr_vel, next_vel);
                    if dot_prod < 0  % Opposite directions
                        reversal_count = reversal_count + 1;
                    end
                end
            end
        end
        
        function reset(obj)
            % reset: Reset extractor (e.g., on new trial)
            obj.hand_position_history = zeros(FeatureExtractor.WINDOW_SIZE, 4);
            obj.hand_velocity_history = zeros(FeatureExtractor.WINDOW_SIZE, 4);
            obj.window_idx = 0;
            obj.retry_count = 0;
        end
    end
end
