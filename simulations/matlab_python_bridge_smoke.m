function matlab_python_bridge_smoke()
% matlab_python_bridge_smoke: Optional smoke test for Python bridge backend.

    repo_root = fileparts(mfilename('fullpath'));
    addpath(fullfile(repo_root, 'src', 'matlab'));

    config = buildABConfig('backend', 'python_bridge');
    cli = HesitationModelCLI(config.python.model_root_dir);
    cli.python_cmd = config.python.python_cmd;

    original_dir = pwd;
    cd(config.python.model_root_dir);
    [status_health, out_health] = system(sprintf('PYTHONPATH=src %s -m hesitation.inference.cli health', config.python.python_cmd));
    cd(original_dir);
    assert(status_health == 0, 'Python bridge health check failed: %s', out_health);

    features = struct( ...
        'mean_hand_speed', 0.04, ...
        'pause_ratio', 0.60, ...
        'progress_delta', 0.01, ...
        'reversal_count', int32(1), ...
        'retry_count', int32(0), ...
        'task_step_id', int32(2), ...
        'human_robot_distance', 0.17);
    [prediction, ~] = inferHesitationPythonBridge(features, config, cli);
    required_fields = {'state', 'state_probabilities', 'future_hesitation_prob', 'future_correction_prob', 'confidence'};
    for idx = 1:numel(required_fields)
        assert(isfield(prediction, required_fields{idx}), 'Missing bridge field: %s', required_fields{idx});
    end
    fprintf('MATLAB Python bridge smoke test passed.\n');
end
