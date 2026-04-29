function [state_prob, future_hesitation_prob, future_correction_prob, classes] = infer_classical(model, X)
% INFER_CLASSICAL Runs inference using the classical model parameters
%   [state_prob, future_hesitation_prob, future_correction_prob, classes] = infer_classical(model, X)
%
%   model is a struct containing the model parameters loaded from JSON.
%   X is an N x 7 matrix of features in the order:
%   ["mean_speed", "speed_variance", "pause_ratio", "direction_changes",
%    "progress_delta", "backtrack_ratio", "mean_workspace_distance"]

    % 1. Standardize features
    means = get_numeric_vector(model.scaler.means)';
    stds = get_numeric_vector(model.scaler.stds)';
    
    X_scaled = (X - means) ./ stds;
    
    % 2. State inference (Multinomial Logistic Regression)
    classes = model.state.classes;
    if ~iscell(classes)
        error('model.state.classes must be a cell array of strings');
    end
    
    num_classes = length(classes);
    num_samples = size(X, 1);
    
    state_logits = zeros(num_samples, num_classes);
    for i = 1:num_classes
        cls = classes{i};
        w = get_numeric_vector(model.state.weights.(cls));
        b = model.state.biases.(cls);
        state_logits(:, i) = X_scaled * w + b;
    end
    
    % Softmax
    exp_logits = exp(state_logits - max(state_logits, [], 2));
    state_prob = exp_logits ./ sum(exp_logits, 2);
    
    % 3. Future hesitation inference (Binary Logistic Regression)
    w_hesitation = get_numeric_vector(model.future_hesitation.weights);
    b_hesitation = model.future_hesitation.bias;
    hesitation_logits = X_scaled * w_hesitation + b_hesitation;
    future_hesitation_prob = 1 ./ (1 + exp(-hesitation_logits));
    
    % 4. Future correction inference (Binary Logistic Regression)
    w_correction = get_numeric_vector(model.future_correction.weights);
    b_correction = model.future_correction.bias;
    correction_logits = X_scaled * w_correction + b_correction;
    future_correction_prob = 1 ./ (1 + exp(-correction_logits));
end

function val = get_numeric_vector(data)
% Helper to handle both Octave/MATLAB parsing quirks (cell vs numeric)
    if iscell(data)
        val = cell2mat(data);
    else
        val = data;
    end
    val = reshape(val, [], 1);
end
