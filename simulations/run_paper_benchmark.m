function run_paper_benchmark()
% run_paper_benchmark  Executes the canonical A/B benchmarking matrix across
%   multiple seeds, aggregates the metrics, computes statistics, and generates
%   paper-ready figures and tables.
%
%   Outputs are written to:
%     artifacts/paper_results/tables/   (CSV)
%     artifacts/paper_results/figures/  (PNG)

    %% Configuration
    rng(42, 'twister');                          % reproducible seed list
    NUM_SEEDS   = 500;
    seeds       = randi([1, 100000], 1, NUM_SEEDS);
    num_seeds   = length(seeds);

    root_dir    = fileparts(fileparts(mfilename('fullpath')));
    output_root = fullfile(root_dir, 'artifacts', 'paper_results');
    dirs.tables  = fullfile(output_root, 'tables');
    dirs.figures = fullfile(output_root, 'figures');

    if ~exist(dirs.tables,  'dir'), mkdir(dirs.tables);  end
    if ~exist(dirs.figures, 'dir'), mkdir(dirs.figures); end

    fprintf('=== IEOM Paper Benchmark Runner ===\n');
    fprintf('Running %d seeds across all scenarios...\n\n', num_seeds);

    all_baseline_rows    = struct([]);
    all_aware_rows       = struct([]);
    all_comparison_rows  = struct([]);

    % Source-usage tracking (how often each inference path fires)
    source_counts = struct( ...
        'native_model',      0, ...
        'error_fallback',    0, ...
        'stub_fallback',     0, ...
        'script_passthrough', 0);

    %% Main loop
    for i = 1:num_seeds
        current_seed = seeds(i);
        fprintf('--- Running Seed %d/%d (Value: %d) ---\n', i, num_seeds, current_seed);

        % enable_replay=false speeds up batch execution (no determinism check needed)
        summary = stage3_run_ab_scenarios( ...
            'deterministic_seed', current_seed, ...
            'enable_replay',      false);

        % --- Aggregate baseline rows ---
        for r = 1:height(summary.baseline_metrics)
            row      = table2struct(summary.baseline_metrics(r, :));
            row.seed = current_seed;
            if isempty(all_baseline_rows)
                all_baseline_rows = row;
            else
                all_baseline_rows(end + 1) = row; %#ok<AGROW>
            end
        end

        % --- Aggregate hesitation-aware rows ---
        for r = 1:height(summary.hesitation_aware_metrics)
            row      = table2struct(summary.hesitation_aware_metrics(r, :));
            row.seed = current_seed;
            if isempty(all_aware_rows)
                all_aware_rows = row;
            else
                all_aware_rows(end + 1) = row; %#ok<AGROW>
            end
        end

        % --- Aggregate comparison rows ---
        for r = 1:height(summary.comparison)
            row      = table2struct(summary.comparison(r, :));
            row.seed = current_seed;
            if isempty(all_comparison_rows)
                all_comparison_rows = row;
            else
                all_comparison_rows(end + 1) = row; %#ok<AGROW>
            end
        end

        % --- Scan JSONL logs to tally inference source usage ---
        scenarios = summary.scenario_names;
        if ischar(scenarios)
            scenarios = {scenarios};   % scalar safety
        end
        for s = 1:length(scenarios)
            log_path = fullfile(summary.feature_log_dir, ...
                sprintf('%s_hesitation_aware.jsonl', scenarios{s}));
            if exist(log_path, 'file')
                log_rows = readJsonl(log_path);
                for r_idx = 1:length(log_rows)
                    src = char(log_rows(r_idx).source);
                    if isfield(source_counts, src)
                        source_counts.(src) = source_counts.(src) + 1;
                    else
                        source_counts.(src) = 1;
                    end
                end
            end
        end
    end  % seed loop

    fprintf('\nSimulation runs complete. Generating aggregate artifacts...\n');

    %% 1. Build aggregate tables
    baseline_tbl    = struct2table(all_baseline_rows);
    aware_tbl       = struct2table(all_aware_rows);
    comparison_tbl  = struct2table(all_comparison_rows);

    % Main A/B summary (mean, std, CI, Cohen's d, Wilcoxon p)
    main_summary = createMainSummary(baseline_tbl, aware_tbl);
    writetable(main_summary, fullfile(dirs.tables, 'main_ab_benchmark_summary.csv'));

    % Per-scenario breakdown
    per_scenario_tbl = createPerScenarioSummary(baseline_tbl, aware_tbl);
    writetable(per_scenario_tbl, fullfile(dirs.tables, 'per_scenario_summary.csv'));

    % Inference source usage
    source_tbl = struct2table(source_counts);
    writetable(source_tbl, fullfile(dirs.tables, 'source_usage_summary.csv'));

    % Conflict-level stratified summary
    stratified_tbl = createStratifiedSummary(baseline_tbl, aware_tbl);
    writetable(stratified_tbl, fullfile(dirs.tables, 'environment_stratified_summary.csv'));

    % Failure / underperformance analysis
    failure_tbl = createFailureAnalysis(aware_tbl, comparison_tbl);
    writetable(failure_tbl, fullfile(dirs.tables, 'failure_analysis_report.csv'));

    %% 2. Generate figures
    generateFigures(baseline_tbl, aware_tbl, dirs.figures);

    fprintf('=== Paper Artifact Generation Complete ===\n');
    fprintf('Results saved to: %s\n', output_root);
end

function sum_tbl = createMainSummary(base, aware)
% createMainSummary  Aggregate A/B statistics with CIs and non-parametric tests.
%
%   For each metric the table reports:
%     policy_a_mean / _std        — Policy A (baseline) mean and std
%     policy_b_mean / _std        — Policy B (hesitation-aware) mean and std
%     ci_95_halfwidth_a/b         — 95% CI half-width (z=1.96, assuming CLT)
%     absolute_improvement        — mean(A) - mean(B)  [positive = B wins]
%     percent_improvement         — improvement as % of policy A mean
%     effect_size_d               — Cohen's d (pooled-std)
%     wilcoxon_p                  — Wilcoxon signed-rank p-value on paired diffs
%                                   (non-parametric; appropriate for event counts)

    metrics = { ...
        'task_completion_time_sec', ...
        'overlap_risk_event_count', ...
        'robot_hold_count', ...
        'human_wait_time_sec', ...
        'unnecessary_slowdown_count'};

    res = struct();
    for i = 1:length(metrics)
        m = metrics{i};
        a_vals = base.(m);
        b_vals = aware.(m);
        n      = length(a_vals);

        a_mean = mean(a_vals);
        a_std  = std(a_vals);
        b_mean = mean(b_vals);
        b_std  = std(b_vals);
        abs_imp = a_mean - b_mean;

        if a_mean > 0
            pct_imp = (abs_imp / a_mean) * 100;
        else
            pct_imp = 0;
        end

        pooled_std  = sqrt((a_std^2 + b_std^2) / 2 + eps);
        cohen_d     = abs_imp / pooled_std;

        % 95% CI half-width via CLT (symmetric, z = 1.96)
        ci_a = 1.96 * a_std / sqrt(n);
        ci_b = 1.96 * b_std / sqrt(n);

        % Wilcoxon signed-rank test on per-run paired differences (A - B)
        % signrank() requires the Statistics Toolbox; fall back gracefully.
        diffs = a_vals - b_vals;
        try
            p_wilcox = signrank(diffs);
        catch
            p_wilcox = NaN;  % Stats Toolbox not available
        end

        res(i).metric              = string(m);
        res(i).n_observations      = n;
        res(i).policy_a_mean       = a_mean;
        res(i).policy_a_std        = a_std;
        res(i).ci_95_halfwidth_a   = ci_a;
        res(i).policy_b_mean       = b_mean;
        res(i).policy_b_std        = b_std;
        res(i).ci_95_halfwidth_b   = ci_b;
        res(i).absolute_improvement = abs_imp;
        res(i).percent_improvement  = pct_imp;
        res(i).effect_size_d        = cohen_d;
        res(i).wilcoxon_p           = p_wilcox;
    end
    sum_tbl = struct2table(res);
end

function sc_tbl = createPerScenarioSummary(base, aware)
    scenarios = unique(base.scenario);
    res = struct();
    idx = 1;
    for i = 1:length(scenarios)
        sc = scenarios(i);
        b_sc = base(strcmp(base.scenario, sc), :);
        a_sc = aware(strcmp(aware.scenario, sc), :);
        
        metrics = {'task_completion_time_sec', 'overlap_risk_event_count'};
        for m = 1:length(metrics)
            met = metrics{m};
            res(idx).scenario = string(sc);
            res(idx).metric = string(met);
            res(idx).policy_a_mean = mean(b_sc.(met));
            res(idx).policy_b_mean = mean(a_sc.(met));
            res(idx).improvement = mean(b_sc.(met)) - mean(a_sc.(met));
            idx = idx + 1;
        end
    end
    sc_tbl = struct2table(res);
end

function fail_tbl = createFailureAnalysis(aware, comp)
    % A run is considered a failure/underperformance if completion time delta > 0
    % or overlap risk delta > 0 (meaning Aware was worse than Baseline)
    underperformed = comp(comp.task_completion_time_delta > 0 | comp.overlap_risk_event_delta > 0, :);
    
    res = struct();
    for i = 1:height(underperformed)
        res(i).scenario = string(underperformed.scenario{i});
        res(i).seed = underperformed.seed(i);
        res(i).time_penalty_sec = underperformed.task_completion_time_delta(i);
        res(i).extra_overlap_events = underperformed.overlap_risk_event_delta(i);
        
        % Add mismatch count from aware_tbl
        match_idx = find(strcmp(aware.scenario, underperformed.scenario{i}) & aware.seed == underperformed.seed(i));
        if ~isempty(match_idx)
            res(i).prediction_mismatch_count = aware.prediction_mismatch_count(match_idx(1));
        else
            res(i).prediction_mismatch_count = -1;
        end
    end
    
    if isempty(fieldnames(res))
        fail_tbl = table('Size', [0 5], 'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
            'VariableNames', {'scenario', 'seed', 'time_penalty_sec', 'extra_overlap_events', 'prediction_mismatch_count'});
    else
        fail_tbl = struct2table(res);
    end
end

function generateFigures(base, aware, out_dir)
    scenarios = unique(base.scenario);
    num_scenarios = length(scenarios);
    
    % Ensure figures don't pop up
    f1 = figure('Visible', 'off');
    f2 = figure('Visible', 'off');
    f3 = figure('Visible', 'off');
    f4 = figure('Visible', 'off');
    f5 = figure('Visible', 'off');
    
    %% 1. Safety Comparison
    set(0, 'CurrentFigure', f1);
    base_overlap = zeros(1, num_scenarios);
    aware_overlap = zeros(1, num_scenarios);
    for i = 1:num_scenarios
        base_overlap(i) = sum(base.overlap_risk_event_count(strcmp(base.scenario, scenarios(i))));
        aware_overlap(i) = sum(aware.overlap_risk_event_count(strcmp(aware.scenario, scenarios(i))));
    end
    
    bar([base_overlap', aware_overlap']);
    title('Safety Comparison: Total Overlap Events');
    xticklabels(strrep(cellstr(scenarios), '_', '\_'));
    xtickangle(25);
    ylabel('Event Count');
    legend('Policy A (Baseline)', 'Policy B (Model-Aware)', 'Location', 'northwest');
    grid on;
    saveas(f1, fullfile(out_dir, 'safety_comparison.png'));
    
    %% 2. Efficiency Comparison
    set(0, 'CurrentFigure', f2);
    base_time = zeros(1, num_scenarios);
    aware_time = zeros(1, num_scenarios);
    for i = 1:num_scenarios
        base_time(i) = mean(base.task_completion_time_sec(strcmp(base.scenario, scenarios(i))));
        aware_time(i) = mean(aware.task_completion_time_sec(strcmp(aware.scenario, scenarios(i))));
    end
    
    bar([base_time', aware_time']);
    title('Efficiency Comparison: Mean Task Completion Time');
    xticklabels(strrep(cellstr(scenarios), '_', '\_'));
    xtickangle(25);
    ylabel('Time (s)');
    legend('Policy A (Baseline)', 'Policy B (Model-Aware)', 'Location', 'northwest');
    grid on;
    saveas(f2, fullfile(out_dir, 'efficiency_comparison.png'));
    
    %% 3. Tradeoff Plot
    set(0, 'CurrentFigure', f3);
    hold on;
    colors = lines(num_scenarios);
    h_leg = [];
    leg_names = {};
    for i = 1:num_scenarios
        idx_base = strcmp(base.scenario, scenarios(i));
        idx_aware = strcmp(aware.scenario, scenarios(i));
        
        h1 = scatter(base.overlap_risk_event_count(idx_base), base.task_completion_time_sec(idx_base), 50, colors(i,:), 'o', 'filled', 'MarkerFaceAlpha', 0.6);
        h2 = scatter(aware.overlap_risk_event_count(idx_aware), aware.task_completion_time_sec(idx_aware), 50, colors(i,:), '^', 'filled', 'MarkerFaceAlpha', 0.6);
        
        h_leg(end+1) = h1; %#ok<AGROW>
        h_leg(end+1) = h2; %#ok<AGROW>
        leg_names{end+1} = sprintf('%s (Policy A)', strrep(char(scenarios(i)), '_', '\_')); %#ok<AGROW>
        leg_names{end+1} = sprintf('%s (Policy B)', strrep(char(scenarios(i)), '_', '\_')); %#ok<AGROW>
    end
    title('Tradeoff: Safety vs Efficiency across all runs');
    xlabel('Overlap Risk Events (Lower is safer)');
    ylabel('Task Completion Time (Lower is more efficient)');
    legend(h_leg, leg_names, 'Location', 'bestoutside');
    grid on;
    % Make figure a bit wider to accommodate the outside legend
    set(f3, 'Position', [100, 100, 800, 500]);
    saveas(f3, fullfile(out_dir, 'tradeoff_scatter.png'));
    
    %% 4. Robot Hold Count
    set(0, 'CurrentFigure', f4);
    base_holds = zeros(1, num_scenarios);
    aware_holds = zeros(1, num_scenarios);
    for i = 1:num_scenarios
        base_holds(i) = mean(base.robot_hold_count(strcmp(base.scenario, scenarios(i))));
        aware_holds(i) = mean(aware.robot_hold_count(strcmp(aware.scenario, scenarios(i))));
    end
    
    bar([base_holds', aware_holds']);
    title('Intervention Comparison: Mean Robot Hold Count');
    xticklabels(strrep(cellstr(scenarios), '_', '\_'));
    xtickangle(25);
    ylabel('Mean Holds per Run');
    legend('Policy A (Baseline)', 'Policy B (Model-Aware)', 'Location', 'northwest');
    grid on;
    saveas(f4, fullfile(out_dir, 'intervention_comparison.png'));
    
    %% 5. Human Wait Time
    set(0, 'CurrentFigure', f5);
    base_wait = zeros(1, num_scenarios);
    aware_wait = zeros(1, num_scenarios);
    for i = 1:num_scenarios
        base_wait(i) = mean(base.human_wait_time_sec(strcmp(base.scenario, scenarios(i))));
        aware_wait(i) = mean(aware.human_wait_time_sec(strcmp(aware.scenario, scenarios(i))));
    end
    
    bar([base_wait', aware_wait']);
    title('Efficiency Comparison: Mean Human Wait Time');
    xticklabels(strrep(cellstr(scenarios), '_', '\_'));
    xtickangle(25);
    ylabel('Wait Time (s)');
    legend('Policy A (Baseline)', 'Policy B (Model-Aware)', 'Location', 'northwest');
    grid on;
    saveas(f5, fullfile(out_dir, 'human_wait_time_comparison.png'));
    
    close(f1); close(f2); close(f3); close(f4); close(f5);
end

function strat_tbl = createStratifiedSummary(base, aware)
    low_conflict_envs = {'low_conflict_open'};
    high_conflict_envs = {'narrow_assembly_bench', 'precision_insertion', 'inspection_rework', 'shared_bin_access'};
    
    idx_base_low = ismember(base.scenario, low_conflict_envs);
    idx_aware_low = ismember(aware.scenario, low_conflict_envs);
    
    idx_base_high = ismember(base.scenario, high_conflict_envs);
    idx_aware_high = ismember(aware.scenario, high_conflict_envs);
    
    metrics = {'task_completion_time_sec', 'overlap_risk_event_count', 'robot_hold_count'};
    res = struct();
    
    % Low conflict
    for m = 1:length(metrics)
        met = metrics{m};
        res(m).conflict_level = string('Low Conflict');
        res(m).metric = string(met);
        res(m).policy_a_mean = mean(base.(met)(idx_base_low));
        res(m).policy_b_mean = mean(aware.(met)(idx_aware_low));
        res(m).improvement = res(m).policy_a_mean - res(m).policy_b_mean;
    end
    
    % High conflict
    offset = length(metrics);
    for m = 1:length(metrics)
        met = metrics{m};
        res(offset + m).conflict_level = string('High Conflict');
        res(offset + m).metric = string(met);
        res(offset + m).policy_a_mean = mean(base.(met)(idx_base_high));
        res(offset + m).policy_b_mean = mean(aware.(met)(idx_aware_high));
        res(offset + m).improvement = res(offset + m).policy_a_mean - res(offset + m).policy_b_mean;
    end
    
    strat_tbl = struct2table(res);
end

function rows = readJsonl(path)
    fid = fopen(path, 'r');
    if fid < 0
        error('Cannot open file for reading: %s', path);
    end
    cleaner = onCleanup(@() fclose(fid));

    raw_rows = cell(1000, 1);
    idx = 1;
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        if isempty(strtrim(line))
            continue;
        end
        raw_rows{idx} = jsondecode(line);
        idx = idx + 1;
        if idx > numel(raw_rows)
            raw_rows = [raw_rows; cell(1000, 1)]; %#ok<AGROW>
        end
    end
    raw_rows = raw_rows(1:idx-1);
    if isempty(raw_rows)
        rows = struct([]);
    else
        rows = [raw_rows{:}];
    end
end
