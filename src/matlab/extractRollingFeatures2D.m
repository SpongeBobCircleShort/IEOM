function [features, buffers] = extractRollingFeatures2D(buffers, human, robot, interaction, task_restart_flag, config)
% extractRollingFeatures2D: Maintain rolling buffers and compute 7-feature inference input.

    buffers.human_pos_xy = appendWindowRow(buffers.human_pos_xy, human.pos_xy, config.window_size_frames);
    buffers.human_speed = appendWindowScalar(buffers.human_speed, norm(human.vel_xy), config.window_size_frames);
    buffers.progress = appendWindowScalar(buffers.progress, human.progress_01, config.window_size_frames);
    buffers.separation = appendWindowScalar(buffers.separation, interaction.separation_m, config.window_size_frames);
    buffers.task_restart_flag = appendWindowScalar( ...
        buffers.task_restart_flag, ...
        double(task_restart_flag), ...
        config.window_size_frames);

    current_speed = buffers.human_speed(end);
    features = struct( ...
        'mean_hand_speed', mean(buffers.human_speed), ...
        'pause_ratio', sum(buffers.human_speed < 0.03) / numel(buffers.human_speed), ...
        'progress_delta', max(buffers.progress(end) - buffers.progress(1), 0.0), ...
        'reversal_count', int32(countProgressReversals(buffers.progress)), ...
        'retry_count', int32(human.rework_count), ...
        'task_step_id', int32(max(0, min(3, human.target_idx - 1))), ...
        'human_robot_distance', interaction.separation_m);

    if isnan(features.mean_hand_speed)
        features.mean_hand_speed = current_speed;
    end
end

function values = appendWindowRow(values, row, max_size)
    if isempty(values)
        values = row;
    else
        values = [values; row]; %#ok<AGROW>
    end
    if size(values, 1) > max_size
        values = values((end - max_size + 1):end, :);
    end
end

function values = appendWindowScalar(values, scalar_value, max_size)
    if isempty(values)
        values = scalar_value;
    else
        values = [values; scalar_value]; %#ok<AGROW>
    end
    if numel(values) > max_size
        values = values((end - max_size + 1):end);
    end
end

function reversal_count = countProgressReversals(progress_values)
    reversal_count = 0;
    if numel(progress_values) < 3
        return;
    end

    deltas = diff(progress_values);
    for idx = 2:numel(deltas)
        previous = deltas(idx - 1);
        current = deltas(idx);
        if abs(previous) < 1e-6 || abs(current) < 1e-6
            continue;
        end
        if sign(previous) ~= sign(current)
            reversal_count = reversal_count + 1;
        end
    end
end
