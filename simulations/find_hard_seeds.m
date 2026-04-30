% find_hard_seeds.m
% Search for seeds that cause Policy B to have an overlap event in each scenario
clear; clc;

% Need to use the same config setup as stage3_run_ab_scenarios
opts = struct('deterministic_seed', 1, 'use_stub', false, 'enable_replay', false, 'window_size', 12);
root_dir = fileparts(pwd);
config = struct();
config.dt_sec = 0.1;
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

model_path = fullfile(root_dir, 'simulations', 'classical_model.json');
raw_json = fileread(model_path);
config.trained_model = jsondecode(raw_json);

envs = {
    struct('name', 'low_conflict_open', 'weights', [0.70, 0.05, 0.05, 0.05, 0.10, 0.05], 'task_step_count', 6, 'shared_zone_x', [0.45, 0.55], 'overlap_buffer', 0.05, 'conflict_level', 'low'), ...
    struct('name', 'narrow_assembly_bench', 'weights', [0.40, 0.15, 0.10, 0.05, 0.10, 0.20], 'task_step_count', 6, 'shared_zone_x', [0.30, 0.70], 'overlap_buffer', 0.10, 'conflict_level', 'high'), ...
    struct('name', 'precision_insertion', 'weights', [0.20, 0.25, 0.25, 0.15, 0.10, 0.05], 'task_step_count', 6, 'shared_zone_x', [0.48, 0.52], 'overlap_buffer', 0.15, 'conflict_level', 'high'), ...
    struct('name', 'inspection_rework', 'weights', [0.20, 0.10, 0.10, 0.40, 0.10, 0.10], 'task_step_count', 6, 'shared_zone_x', [0.40, 0.60], 'overlap_buffer', 0.08, 'conflict_level', 'high')
};

% We need the functions from stage3_run_ab_scenarios.m
% Let's just modify stage3_run_ab_scenarios to run until it finds an overlap.
