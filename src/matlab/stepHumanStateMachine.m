function [human, event_flags] = stepHumanStateMachine(human, robot, interaction, schedule, config, frame_idx)
% stepHumanStateMachine: Update human state and 2D kinematics for one frame.

    event_flags = struct('task_restart', false, 'hesitation_onset', false);
    human.state_age_frames = human.state_age_frames + 1;

    mild_active = frame_idx >= schedule.hesitation_onset_frame && frame_idx <= schedule.hesitation_end_frame;
    strong_active = schedule.escalate_to_strong && ...
        frame_idx >= schedule.strong_onset_frame && ...
        frame_idx <= schedule.hesitation_end_frame;

    crowding_mild = human.crowding_frames_mild >= config.human.crowding_mild_frames;
    crowding_strong = human.crowding_frames_strong >= config.human.crowding_strong_frames;
    stable_release = human.stable_release_frames >= config.human.stable_release_frames;

    if schedule.rework_trigger_frame > 0 && ...
            frame_idx == schedule.rework_trigger_frame && ...
            ~human.rework_triggered
        human.rework_triggered = true;
        human.rework_count = human.rework_count + 1;
        human.resume_target_idx = human.target_idx;
        human.state = 'correction_rework';
        human.state_age_frames = 0;
        human.target_idx = max(1, human.target_idx - 1);
        event_flags.task_restart = true;
        event_flags.hesitation_onset = true;
    elseif strcmp(human.state, 'normal_progress')
        if mild_active || crowding_mild
            human.state = 'mild_hesitation';
            human.state_age_frames = 0;
            event_flags.hesitation_onset = true;
        end
    elseif strcmp(human.state, 'mild_hesitation')
        if strong_active || crowding_strong
            human.state = 'strong_hesitation';
            human.state_age_frames = 0;
        elseif stable_release && interaction.separation_m > 0.25 && human.progress_trend_positive
            human.state = 'normal_progress';
            human.state_age_frames = 0;
        end
    elseif strcmp(human.state, 'strong_hesitation')
        if human.stable_release_frames >= config.human.stable_release_frames && ~interaction.robot_in_shared_zone
            human.state = 'mild_hesitation';
            human.state_age_frames = 0;
        end
    elseif strcmp(human.state, 'correction_rework')
        if reachedWaypoint(human.pos_xy, config.workspace.human_waypoints_xy(human.target_idx, :), config.human.waypoint_tolerance_m)
            human.target_idx = max(human.resume_target_idx, human.target_idx + 1);
            human.state = 'normal_progress';
            human.state_age_frames = 0;
        end
    end

    speed_scale = 1.0;
    pause_now = false;
    if strcmp(human.state, 'mild_hesitation')
        speed_scale = config.human.mild_speed_scale;
        pause_now = mod(frame_idx, config.human.pause_every_n_frames) == 0;
        human.mild_frames = human.mild_frames + 1;
    elseif strcmp(human.state, 'strong_hesitation')
        speed_scale = config.human.strong_speed_scale;
        pause_now = mod(frame_idx, 3) ~= 0;
        human.strong_frames = human.strong_frames + 1;
    elseif strcmp(human.state, 'correction_rework')
        speed_scale = 0.60;
    elseif strcmp(human.state, 'done')
        speed_scale = 0.0;
    end

    if pause_now
        commanded_speed = 0.0;
    else
        commanded_speed = schedule.human_nominal_speed_mps * speed_scale;
    end

    target = config.workspace.human_waypoints_xy(human.target_idx, :);
    if strcmp(human.state, 'correction_rework')
        target = config.workspace.human_waypoints_xy(max(1, human.target_idx), :);
    end

    [new_pos_xy, new_vel_xy] = moveTowardTarget(human.pos_xy, target, commanded_speed, config.dt_sec);
    new_pos_xy = new_pos_xy + deterministicJitter(schedule.seed, frame_idx, schedule.jitter_scale_m);
    new_pos_xy = clampToBounds(new_pos_xy, config.workspace.bounds_xy);

    if strcmp(human.state, 'strong_hesitation') && pause_now
        new_vel_xy = [0.0, 0.0];
    elseif strcmp(human.state, 'strong_hesitation')
        new_vel_xy = new_vel_xy * config.human.strong_micro_step_scale;
    end

    human.pos_xy = new_pos_xy;
    human.vel_xy = new_vel_xy;
    human.progress_01 = computeHumanProgress(human.pos_xy, human.target_idx, config.workspace);
    human.progress_trend_positive = human.progress_01 >= human.last_progress_01 - 1e-6;
    human.last_progress_01 = human.progress_01;

    if norm(new_vel_xy) < 1e-6
        human.wait_frames = human.wait_frames + 1;
    end

    if strcmp(human.state, 'normal_progress') || strcmp(human.state, 'mild_hesitation') || strcmp(human.state, 'strong_hesitation')
        human = advanceHumanTarget(human, config.workspace, config.human.waypoint_tolerance_m);
        if human.target_idx >= size(config.workspace.human_waypoints_xy, 1) && ...
                reachedWaypoint(human.pos_xy, config.workspace.human_waypoints_xy(end, :), config.human.waypoint_tolerance_m)
            human.state = 'done';
            human.vel_xy = [0.0, 0.0];
        end
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

function human = advanceHumanTarget(human, workspace, tolerance_m)
    while human.target_idx < size(workspace.human_waypoints_xy, 1) && ...
            reachedWaypoint(human.pos_xy, workspace.human_waypoints_xy(human.target_idx, :), tolerance_m)
        human.target_idx = human.target_idx + 1;
    end
end

function flag = reachedWaypoint(pos_xy, target_xy, tolerance_m)
    flag = norm(pos_xy - target_xy) <= tolerance_m;
end

function progress = computeHumanProgress(pos_xy, target_idx, workspace)
    total_distance = 0.0;
    for idx = 1:(size(workspace.human_waypoints_xy, 1) - 1)
        total_distance = total_distance + norm(workspace.human_waypoints_xy(idx + 1, :) - workspace.human_waypoints_xy(idx, :));
    end

    remaining = norm(workspace.human_waypoints_xy(target_idx, :) - pos_xy);
    for idx = target_idx:(size(workspace.human_waypoints_xy, 1) - 1)
        remaining = remaining + norm(workspace.human_waypoints_xy(idx + 1, :) - workspace.human_waypoints_xy(idx, :));
    end
    progress = max(0.0, min(1.0, 1.0 - (remaining / max(total_distance, 1e-6))));
end

function jitter = deterministicJitter(seed, frame_idx, scale)
    phase_a = sin((seed * 0.137) + frame_idx * 0.41);
    phase_b = cos((seed * 0.271) + frame_idx * 0.29);
    jitter = scale * [phase_a, phase_b];
end

function pos_xy = clampToBounds(pos_xy, bounds_xy)
    pos_xy(1) = min(max(pos_xy(1), bounds_xy(1, 1)), bounds_xy(1, 2));
    pos_xy(2) = min(max(pos_xy(2), bounds_xy(2, 1)), bounds_xy(2, 2));
end
