% Quick diagnostic - run 1 scenario to see if hesitation-aware works
clear; clc;

fprintf('Running enhanced diagnostic...\n\n');

global STATE_COUNTS;
STATE_COUNTS = struct();

% Run just narrow_assembly_bench with both policies
results_baseline = stage3_run_ab_scenarios('narrow_assembly_bench', 'baseline', 1);
results_aware = stage3_run_ab_scenarios('narrow_assembly_bench', 'hesitation_aware', 1);

fprintf('\n=== DIAGNOSTIC RESULTS ===\n');
fprintf('Baseline overlaps: %.2f\n', results_baseline.overlap_count);
fprintf('Hesitation-aware overlaps: %.2f\n', results_aware.overlap_count);
fprintf('Baseline holds: %d\n', results_baseline.hold_count);
fprintf('Hesitation-aware holds: %d\n', results_aware.hold_count);

if results_baseline.overlap_count == results_aware.overlap_count && ...
   results_baseline.hold_count == results_aware.hold_count
    fprintf('\nWARNING: Policies are IDENTICAL - hesitation-aware is not active!\n');
else
    fprintf('\nPolicies differ - hesitation-aware is active\n');
end

fprintf('\n=== STATE DISTRIBUTION ===\n');
state_names = fieldnames(STATE_COUNTS);
if ~isempty(state_names)
    total = sum(structfun(@(x) x, STATE_COUNTS));
    for i = 1:length(state_names)
        count = STATE_COUNTS.(state_names{i});
        fprintf('%30s: %4d (%.1f%%)\n', state_names{i}, count, 100*count/total);
    end
else
    fprintf('No state counts captured!\n');
end
