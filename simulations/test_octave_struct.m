% Test Octave struct access
clear; clc;

% Create struct with cell arrays
robot_speeds = struct( ...
    'name',  {'Slow', 'Moderate', 'Aggressive'}, ...
    'speed', {0.4,    0.9,        1.8          }  ...
);

fprintf('Testing struct access:\n');

% Test different access methods
try
    fprintf('Method 1: robot_speeds.speed{1}\n');
    v1 = robot_speeds.speed{1};
    fprintf('  Success: %.2f\n', v1);
catch e
    fprintf('  Failed: %s\n', e.message);
end

try
    fprintf('Method 2: robot_speeds.name(1)\n');
    n1 = robot_speeds.name{1};
    fprintf('  Success: %s\n', n1);
catch e
    fprintf('  Failed: %s\n', e.message);
end

% Alternative: use regular arrays
robot_names = {'Slow', 'Moderate', 'Aggressive'};
robot_speeds_array = [0.4, 0.9, 1.8];

fprintf('\nAlternative method:\n');
for s = 1:3
    fprintf('Scenario %d: %s, v = %.2f\n', s, robot_names{s}, robot_speeds_array(s));
end
