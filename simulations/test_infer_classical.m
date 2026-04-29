% test_infer_classical.m
% Demonstrates how to load the JSON model and call infer_classical.m

% 1. Load the model from JSON
fid = fopen('classical_model.json', 'r');
if fid == -1
    error('Cannot open classical_model.json. Make sure you are in the correct directory.');
end
raw_json = fread(fid, '*char')';
fclose(fid);
model = jsondecode(raw_json);

% 2. Create dummy feature data
% Order: ["mean_speed", "speed_variance", "pause_ratio", "direction_changes", 
%         "progress_delta", "backtrack_ratio", "mean_workspace_distance"]
% Let's create two dummy samples
X = [
    0.1, 0.001, 0.05, 5, 0.02, 0.2, 0.5;
    0.0, 0.0,   0.9,  0, 0.0,  0.0, 0.8
];

% 3. Run inference
[state_prob, fut_hes, fut_corr, classes] = infer_classical(model, X);

% 4. Display results
disp('Classes:');
disp(classes');

disp('State Probabilities:');
disp(state_prob);

disp('Future Hesitation Probability:');
disp(fut_hes);

disp('Future Correction Probability:');
disp(fut_corr);
