function report = matlab_baseline_expected_check()
% Octave/MATLAB-compatible baseline verification without table dependencies.
%
% This script reproduces the core kinematics in baseline_handoff_simulation.m
% and checks whether outputs match expected behavior.

    x_handoff = 5.0;
    x_robot_0 = 10.0;
    x_human_0 = 0.0;
    v_human = 0.5;
    dt = 0.01;
    t_max = 30.0;
    safety_threshold = 0.5;
    iso_compliance_speed = 1.0;

    scenario_names = {'Slow', 'Moderate', 'Aggressive'};
    robot_speeds = [0.4, 0.9, 1.8];

    % Expected from current simulation logic (both-moving min separation).
    expected_task_complete = [12.5, 10.0, 10.0];
    expected_iso = [true, true, false];
    expected_min_sep = [1.0, 2.23, 3.629];

    rows = cell(1, numel(robot_speeds));
    pass_all = true;

    for i = 1:numel(robot_speeds)
        v_robot = robot_speeds(i);
        t_robot_arrive = abs(x_robot_0 - x_handoff) / v_robot;
        t_human_arrive = abs(x_handoff - x_human_0) / v_human;
        t_complete = max(t_robot_arrive, t_human_arrive);

        t_end = min(t_complete + 2.0, t_max);
        t = 0:dt:t_end;

        x_robot = max(x_handoff, x_robot_0 - v_robot .* t);
        x_human = min(x_handoff, x_human_0 + v_human .* t);
        separation = abs(x_robot - x_human);

        both_moving = (x_robot > x_handoff) & (x_human < x_handoff);
        if any(both_moving)
            min_sep = min(separation(both_moving));
        else
            min_sep = separation(1);
        end

        iso_ok = (v_robot <= iso_compliance_speed);

        task_ok = abs(t_complete - expected_task_complete(i)) <= 1e-9;
        iso_match = (iso_ok == expected_iso(i));
        min_sep_ok = abs(min_sep - expected_min_sep(i)) <= 0.03;
        pass_row = task_ok && iso_match && min_sep_ok;
        pass_all = pass_all && pass_row;

        rows{i} = struct( ...
            'scenario', scenario_names{i}, ...
            'v_robot', v_robot, ...
            'task_complete_s', t_complete, ...
            'min_separation_m', min_sep, ...
            'iso_compliant', iso_ok, ...
            'unsafe_under_threshold', min_sep < safety_threshold, ...
            'checks', struct( ...
                'task_complete_match', task_ok, ...
                'iso_flag_match', iso_match, ...
                'min_separation_match', min_sep_ok, ...
                'row_pass', pass_row ...
            ) ...
        );
    end

    report = struct();
    report.generated_at = datestr(now, 30);
    report.pass_all = pass_all;
    report.note = 'Checks follow current baseline_handoff_simulation both-moving min separation logic.';
    report.rows = [rows{:}];

    out_dir = 'reports/phase3_verification';
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    out_mat = fullfile(out_dir, 'matlab_baseline_expected_report.mat');
    save(out_mat, 'report');

    out_txt = fullfile(out_dir, 'matlab_baseline_expected_report.txt');
    fid = fopen(out_txt, 'w');
    fprintf(fid, 'pass_all=%d\n', report.pass_all);
    fprintf(fid, 'generated_at=%s\n', report.generated_at);
    for i = 1:numel(rows)
        r = rows{i};
        fprintf(fid, '%s|v=%.3f|t=%.3f|min_sep=%.3f|iso=%d|row_pass=%d\n', ...
            r.scenario, r.v_robot, r.task_complete_s, r.min_separation_m, ...
            r.iso_compliant, r.checks.row_pass);
    end
    fclose(fid);

    fprintf('Baseline expected-output check complete. pass_all=%d\n', report.pass_all);
    fprintf('Wrote: %s\n', out_mat);
    fprintf('Wrote: %s\n', out_txt);
end
