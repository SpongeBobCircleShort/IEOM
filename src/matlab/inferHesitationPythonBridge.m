function [prediction, cli_handle] = inferHesitationPythonBridge(features, config, cli_handle)
% inferHesitationPythonBridge: Optional bridge around existing HesitationModelCLI.

    if nargin < 3 || isempty(cli_handle)
        cli_handle = HesitationModelCLI(config.python.model_root_dir);
        cli_handle.python_cmd = config.python.python_cmd;
    end

    prediction = cli_handle.predict_single(features);

    if strcmp(prediction.state, 'ready_for_robot_action')
        prediction.future_hesitation_prob = min(prediction.future_hesitation_prob, 0.20);
        prediction.future_correction_prob = min(prediction.future_correction_prob, 0.15);
    elseif strcmp(prediction.state, 'overlap_risk')
        prediction.future_hesitation_prob = max(prediction.future_hesitation_prob, 0.45);
        if prediction.future_hesitation_prob >= 0.50
            prediction.future_correction_prob = max(prediction.future_correction_prob, 0.20);
        end
    end
end
