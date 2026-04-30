function robot = stepRobotPolicy(robot, human, interaction, prediction, policy_name, schedule, config, frame_idx)
% stepRobotPolicy: Update robot mode and 2D motion for policy A or B.

    robot.state_changed = false;
    base_speed_scale = 1.0;

    if frame_idx < schedule.robot_release_frame
        robot.mode = 'proceed';
        robot.speed_scale_cmd = 0.0;
        robot.wait_frames = robot.wait_frames + 1;
        robot.vel_xy = [0.0, 0.0];
        return;
    elseif interaction.separation_m < config.workspace.hard_safety_radius_m
        robot = setRobotMode(robot, 'emergency_stop');
    elseif strcmp(robot.mode, 'emergency_stop') && interaction.separation_m >= config.workspace.emergency_release_radius_m
        robot = setRobotMode(robot, 'proceed');
    elseif strcmp(policy_name, 'A')
        if ~strcmp(robot.mode, 'emergency_stop')
            robot = setRobotMode(robot, 'proceed');
        end
    else
        robot = applyPolicyB(robot, interaction, prediction, config);
    end

    if strcmp(robot.mode, 'hold') || strcmp(robot.mode, 'emergency_stop')
        base_speed_scale = 0.0;
        robot.wait_frames = robot.wait_frames + 1;
    elseif strcmp(robot.mode, 'slow')
        base_speed_scale = config.policy_b.slow_speed_scale;
        robot.wait_frames = robot.wait_frames + 1;
        robot.slow_frames = robot.slow_frames + 1;
    end

    robot.speed_scale_cmd = base_speed_scale;
    target_idx = robot.target_idx;
    if strcmp(policy_name, 'B') && strcmp(robot.mode, 'hold')
        target_idx = min(2, robot.target_idx);
    elseif strcmp(policy_name, 'B') && strcmp(robot.mode, 'slow') && robot.target_idx >= 3 && ~interaction.robot_in_shared_zone
        target_idx = 2;
    end

    target_xy = config.workspace.robot_waypoints_xy(target_idx, :);
    [new_pos_xy, new_vel_xy] = moveTowardTarget(robot.pos_xy, target_xy, schedule.robot_nominal_speed_mps * base_speed_scale, config.dt_sec);
    robot.pos_xy = clampToBounds(new_pos_xy, config.workspace.bounds_xy);
    robot.vel_xy = new_vel_xy;

    if base_speed_scale > 0.0
        while robot.target_idx < size(config.workspace.robot_waypoints_xy, 1) && ...
                reachedWaypoint(robot.pos_xy, config.workspace.robot_waypoints_xy(robot.target_idx, :), 0.035)
            robot.target_idx = robot.target_idx + 1;
        end
    end

    if robot.target_idx >= size(config.workspace.robot_waypoints_xy, 1) && ...
            reachedWaypoint(robot.pos_xy, config.workspace.robot_waypoints_xy(end, :), 0.035)
        robot.completed = true;
    end
end

function robot = applyPolicyB(robot, interaction, prediction, config)
    [env_hold_threshold, env_slow_threshold, env_release_threshold] = environmentThresholds(config);
    adaptation = adaptiveInterventionAdjustment(config, robot);
    hold_threshold = min(0.95, max(0.20, env_hold_threshold + adaptation));
    slow_threshold = min(0.90, max(0.15, env_slow_threshold + 0.5 * adaptation));
    release_threshold = min(0.60, max(0.05, env_release_threshold + 0.5 * adaptation));

    ambiguity_level = 0.0;
    if isfield(prediction, 'ambiguity_level')
        ambiguity_level = prediction.ambiguity_level;
    end
    if ambiguity_level >= 0.70
        hold_threshold = min(0.96, hold_threshold + 0.08);
        slow_threshold = min(0.92, slow_threshold + 0.05);
    end

    if interaction.robot_in_shared_zone
        robot.release_stable_frames = 0;
        robot = setRobotMode(robot, 'proceed');
        return;
    end

    high_confidence_state = prediction.confidence >= config.policy_b.min_confidence_for_state_hold;
    high_confidence_risk = prediction.confidence >= config.policy_b.min_confidence_for_risk_hold;
    actionable_slow_state = prediction.confidence >= config.policy_b.min_confidence_for_state_slow;

    hold_condition = (high_confidence_state && (strcmp(prediction.state, 'strong_hesitation') || ...
        strcmp(prediction.state, 'correction_rework'))) || ...
        (high_confidence_risk && (prediction.future_hesitation_prob >= hold_threshold || ...
        prediction.future_correction_prob >= config.policy_b.hold_future_correction_threshold));

    slow_condition = (actionable_slow_state && strcmp(prediction.state, 'mild_hesitation')) || ...
        prediction.future_hesitation_prob >= slow_threshold || ...
        interaction.separation_m < config.policy_b.slow_separation_threshold_m;

    low_risk_release = prediction.future_hesitation_prob < release_threshold && ...
        prediction.future_correction_prob < config.policy_b.low_risk_future_correction_threshold && ...
        ~strcmp(prediction.state, 'strong_hesitation') && ...
        ~strcmp(prediction.state, 'correction_rework');

    if hold_condition
        robot.release_stable_frames = 0;
        robot = setRobotMode(robot, 'hold');
        return;
    end

    if slow_condition
        if strcmp(robot.mode, 'hold') || strcmp(robot.mode, 'slow')
            robot.release_stable_frames = 0;
        end
        robot = setRobotMode(robot, 'slow');
        return;
    end

    if strcmp(robot.mode, 'hold') || strcmp(robot.mode, 'slow')
        if low_risk_release
            robot.release_stable_frames = robot.release_stable_frames + 1;
        else
            robot.release_stable_frames = 0;
        end

        if robot.release_stable_frames < config.policy_b.release_required_ticks
            return;
        end
        robot.release_stable_frames = 0;
    end

    robot = setRobotMode(robot, 'proceed');
end

function [hold_value, slow_value, release_value] = environmentThresholds(config)
    hold_value = config.policy_b.hold_future_hesitation_threshold;
    slow_value = config.policy_b.slow_future_hesitation_threshold;
    release_value = config.policy_b.low_risk_future_hesitation_threshold;
    if ~isfield(config, 'env_context') || ~isfield(config, 'environment_thresholds')
        return;
    end
    env_name = config.env_context.environment_name;
    if isfield(config.environment_thresholds, env_name)
        env_cfg = config.environment_thresholds.(env_name);
        if isfield(env_cfg, 'hold_hesitation')
            hold_value = env_cfg.hold_hesitation;
        end
        if isfield(env_cfg, 'slow_hesitation')
            slow_value = env_cfg.slow_hesitation;
        end
        if isfield(env_cfg, 'release_hesitation')
            release_value = env_cfg.release_hesitation;
        end
    end
end

function adjustment = adaptiveInterventionAdjustment(config, robot)
    % Positive adjustment raises thresholds (fewer holds), negative lowers thresholds (more holds).
    adjustment = 0.0;
    if ~isfield(config, 'env_context')
        return;
    end
    conflict_level = max(config.env_context.workspace_constraint_level, config.env_context.task_precision_required);
    hold_rate = robot.hold_count / max(1, robot.wait_frames + robot.hold_count);
    if conflict_level < 0.30
        if hold_rate > 0.18
            adjustment = 0.12;
        else
            adjustment = 0.08;
        end
    elseif conflict_level > 0.70
        adjustment = -0.05;
    end
end

function robot = setRobotMode(robot, new_mode)
    if strcmp(robot.mode, new_mode)
        return;
    end

    robot.mode = new_mode;
    robot.state_changed = true;
    if strcmp(new_mode, 'hold')
        robot.hold_count = robot.hold_count + 1;
    end
end

function [new_pos_xy, new_vel_xy] = moveTowardTarget(pos_xy, target_xy, speed_mps, dt_sec)
    delta = target_xy - pos_xy;
    distance = norm(delta);
    if distance < 1e-9 || speed_mps <= 0.0
        new_pos_xy = pos_xy;
        new_vel_xy = [0.0, 0.0];
        return;
    end

    direction = delta / distance;
    step_distance = min(distance, speed_mps * dt_sec);
    new_vel_xy = direction * (step_distance / dt_sec);
    new_pos_xy = pos_xy + direction * step_distance;
end

function flag = reachedWaypoint(pos_xy, target_xy, tolerance_m)
    flag = norm(pos_xy - target_xy) <= tolerance_m;
end

function pos_xy = clampToBounds(pos_xy, bounds_xy)
    pos_xy(1) = min(max(pos_xy(1), bounds_xy(1, 1)), bounds_xy(1, 2));
    pos_xy(2) = min(max(pos_xy(2), bounds_xy(2, 1)), bounds_xy(2, 2));
end
