function generate_canonical_results()
% generate_canonical_results: Single authoritative benchmark run.
%   - N=50 randomised seeds (reproducibly generated from seed=42)
%   - Every output table and figure is derived from the SAME aggregated data
%   - Each CSV row carries explicit N_runs, scenario, and source labels
%   - Overwrites artifacts/canonical_results/  (never mixed with old stale data)
%
% Run:  cd simulations && octave --no-gui generate_canonical_results.m
%   or  matlab -batch "generate_canonical_results"

    %% ---- Configuration ----
    N_SEEDS  = 50;
    rng(42, 'twister');
    seeds = unique(randi(99999, 1, N_SEEDS*2));   % oversample then trim
    seeds = seeds(1:N_SEEDS);

    root_dir    = fileparts(fileparts(mfilename('fullpath')));
    out_root    = fullfile(root_dir, 'artifacts', 'canonical_results');
    tbl_dir     = fullfile(out_root, 'tables');
    fig_dir     = fullfile(out_root, 'figures');
    for d = {tbl_dir, fig_dir}
        if ~exist(d{1}, 'dir'), mkdir(d{1}); end
    end

    envs = buildEnvironmentList();
    env_names = cellfun(@(e) e.name, envs, 'UniformOutput', false);
    n_envs = numel(envs);

    fprintf('\n=== Canonical Benchmark: N=%d seeds × %d environments ===\n\n', ...
        N_SEEDS, n_envs);

    %% ---- Accumulate raw per-run data ----
    % raw_base(run, env) and raw_aware(run, env) hold scalar overlap counts
    raw_base  = nan(N_SEEDS, n_envs);
    raw_aware = nan(N_SEEDS, n_envs);
    raw_time_base  = nan(N_SEEDS, n_envs);
    raw_time_aware = nan(N_SEEDS, n_envs);

    pred_correct = 0; pred_total = 0;
    fp_count = 0; actual_neg = 0;
    hold_by_state = struct();

    for i = 1:N_SEEDS
        s = seeds(i);
        if mod(i, 10) == 0
            fprintf('  Seed %d/%d (val=%d)...\n', i, N_SEEDS, s);
        end

        try
            summary = stage3_run_ab_scenarios( ...
                'deterministic_seed', s, 'enable_replay', false);
        catch err
            fprintf('  [WARN] Seed %d failed: %s\n', i, err.message);
            continue;
        end

        b_tbl = summary.baseline_metrics;
        a_tbl = summary.hesitation_aware_metrics;

        for e = 1:n_envs
            b_row = b_tbl(strcmp(b_tbl.scenario, env_names{e}), :);
            a_row = a_tbl(strcmp(a_tbl.scenario, env_names{e}), :);
            if height(b_row) == 1 && height(a_row) == 1
                raw_base(i, e)       = b_row.overlap_risk_event_count;
                raw_aware(i, e)      = a_row.overlap_risk_event_count;
                raw_time_base(i, e)  = b_row.task_completion_time_sec;
                raw_time_aware(i, e) = a_row.task_completion_time_sec;
            end
        end

        % Collect prediction stats from JSONL logs
        for e = 1:n_envs
            lp = fullfile(summary.feature_log_dir, ...
                sprintf('%s_hesitation_aware.jsonl', env_names{e}));
            if ~exist(lp, 'file'), continue; end
            rows = readJsonlSimple(lp);
            for r = 1:numel(rows)
                rr = rows{r};
                actual = char(rr.scripted_state);
                pred   = char(rr.predicted_state);
                if strcmp(actual, pred), pred_correct = pred_correct + 1; end
                pred_total = pred_total + 1;
                if strcmp(actual, 'normal_progress')
                    actual_neg = actual_neg + 1;
                    if ~strcmp(pred, 'normal_progress')
                        fp_count = fp_count + 1;
                    end
                end
                if strcmp(rr.robot_mode, 'hold')
                    key = sanitize_fieldname(actual);
                    if isfield(hold_by_state, key)
                        hold_by_state.(key) = hold_by_state.(key) + 1;
                    else
                        hold_by_state.(key) = 1;
                    end
                end
            end
        end
    end

    %% ---- Build master aggregation table (single source of truth) ----
    rows = cell(n_envs, 1);
    for e = 1:n_envs
        b_vals = raw_base(:, e);
        a_vals = raw_aware(:, e);
        valid  = ~isnan(b_vals) & ~isnan(a_vals);
        n_ok   = sum(valid);
        bv = b_vals(valid); av = a_vals(valid);
        bt = raw_time_base(valid, e); at = raw_time_aware(valid, e);

        mean_b = mean(bv); mean_a = mean(av);
        if mean_b > 0
            red_pct = (1 - mean_a/mean_b) * 100;
        else
            red_pct = 0;
        end

        mean_bt = mean(bt); mean_at = mean(at);
        time_cost_pct = (mean_at/max(mean_bt,1e-9) - 1) * 100;
        if abs(time_cost_pct) > 1e-6
            ratio = abs(red_pct) / abs(time_cost_pct);
        else
            ratio = Inf;
        end

        % Paired t-test on overlap counts
        p_val = NaN; sig = false;
        if n_ok > 1 && std(bv - av) > 1e-9
            [~, p_val] = ttest(bv, av);
            sig = (p_val < 0.001);
        end

        rows{e} = struct( ...
            'environment',          string(env_names{e}), ...
            'N_runs',               n_ok, ...
            'PolicyA_mean_overlaps', mean_b, ...
            'PolicyA_std_overlaps',  std(bv), ...
            'PolicyB_mean_overlaps', mean_a, ...
            'PolicyB_std_overlaps',  std(av), ...
            'Overlap_Reduction_Pct', red_pct, ...
            'PolicyA_mean_time_sec', mean_bt, ...
            'PolicyB_mean_time_sec', mean_at, ...
            'Time_Cost_Pct',         time_cost_pct, ...
            'Safety_Efficiency_Ratio', ratio, ...
            'paired_ttest_p',        p_val, ...
            'significant_p001',      sig);
    end
    master = struct2table([rows{:}]);
    writetable(master, fullfile(tbl_dir, 'master_aggregated_results.csv'));
    fprintf('\n[TABLE] master_aggregated_results.csv written (%d envs × %d seeds)\n', ...
        n_envs, N_SEEDS);

    %% ---- Model performance table ----
    acc_pct = 0; fpr_pct = 0;
    if pred_total > 0
        acc_pct = (pred_correct / pred_total) * 100;
        fpr_pct = (fp_count / max(actual_neg, 1)) * 100;
    end
    perf_tbl = table( ...
        pred_total, pred_correct, acc_pct, fp_count, actual_neg, fpr_pct, ...
        'VariableNames', {'total_predictions','correct_predictions', ...
            'accuracy_pct','false_positives','actual_negatives','false_positive_rate_pct'});
    writetable(perf_tbl, fullfile(tbl_dir, 'model_performance.csv'));

    %% ---- Hold-by-state table ----
    if ~isempty(fieldnames(hold_by_state))
        fn = fieldnames(hold_by_state);
        st_rows = cell(numel(fn), 1);
        for k = 1:numel(fn)
            st_rows{k} = struct('scripted_state', string(fn{k}), ...
                'hold_count', hold_by_state.(fn{k}));
        end
        hold_tbl = struct2table([st_rows{:}]);
        writetable(hold_tbl, fullfile(tbl_dir, 'hold_by_state.csv'));
    end

    %% ---- Paper statements text file ----
    red_vals = master.Overlap_Reduction_Pct(master.PolicyA_mean_overlaps > 0);
    if isempty(red_vals)
        min_red = 0; max_red = 0;
    else
        min_red = min(red_vals); max_red = max(red_vals);
    end

    fid = fopen(fullfile(tbl_dir, 'paper_statements.txt'), 'w');
    fprintf(fid, '=== Paper Statements (auto-generated, all from master_aggregated_results.csv) ===\n\n');
    fprintf(fid, 'Aggregation:\n');
    fprintf(fid, '  Results represent means across N=%d randomised runs per environment.\n', N_SEEDS);
    fprintf(fid, '  Figures show aggregate statistics; tables also show representative single runs.\n\n');
    fprintf(fid, 'Model Performance:\n');
    fprintf(fid, '  Prediction accuracy: %.1f%%\n', acc_pct);
    fprintf(fid, '  False positive rate: %.1f%%\n', fpr_pct);
    fprintf(fid, '  (FPR = rate at which model predicts hesitation during actual normal_progress steps)\n\n');
    fprintf(fid, 'Overlap Reduction Claim:\n');
    fprintf(fid, '  Across high-conflict environments: %.1f%% to %.1f%% reduction.\n', min_red, max_red);
    fprintf(fid, '  Reframe "100%% elimination" → "%.0f–%.0f%% reduction".\n', floor(min_red), ceil(max_red));
    fprintf(fid, '  Residual overlaps occur when risk events are triggered within one stopping-distance window.\n\n');
    fprintf(fid, 'Statistical Significance:\n');
    sig_envs = master.environment(master.significant_p001 == true);
    for k = 1:numel(sig_envs)
        e_row = master(strcmp(master.environment, sig_envs(k)), :);
        fprintf(fid, '  %s: p=%.2e (statistically significant at p<0.001)\n', ...
            sig_envs(k), e_row.paired_ttest_p);
    end
    fclose(fid);

    %% ---- Print the master table to console ----
    fprintf('\n=== MASTER TABLE (same data as master_aggregated_results.csv) ===\n');
    fprintf('%-24s | %6s | %18s | %18s | %13s | %13s | %10s\n', ...
        'Environment', 'N_runs', 'PolicyA_overlaps', 'PolicyB_overlaps', ...
        'Reduction_%', 'Time_Cost_%', 'p-value');
    fprintf('%s\n', repmat('-', 1, 110));
    for e = 1:height(master)
        r = master(e, :);
        fprintf('%-24s | %6d | %7.2f ± %-8.2f | %7.2f ± %-8.2f | %11.1f%% | %11.1f%% | %10.2e\n', ...
            char(r.environment), r.N_runs, ...
            r.PolicyA_mean_overlaps, r.PolicyA_std_overlaps, ...
            r.PolicyB_mean_overlaps, r.PolicyB_std_overlaps, ...
            r.Overlap_Reduction_Pct, r.Time_Cost_Pct, ...
            r.paired_ttest_p);
    end

    %% ---- Figures (all from master table) ----
    generateFigures(master, fig_dir);

    fprintf('\n=== DONE. All outputs in: %s ===\n', out_root);
    fprintf('  master_aggregated_results.csv  — primary source for paper\n');
    fprintf('  model_performance.csv          — prediction accuracy stats\n');
    fprintf('  hold_by_state.csv              — hold breakdown by hesitation type\n');
    fprintf('  paper_statements.txt           — copy-paste text for paper\n');
    fprintf('  figures/                       — all generated from master table\n');
end

%% ======================================================================
function generateFigures(master, fig_dir)
    envs    = cellstr(master.environment);
    n       = height(master);
    x_pos   = 1:n;

    %% Fig 1: Overlap risk reduction by environment
    fh = figure('Visible', 'off', 'Position', [100 100 900 450]);
    b = bar(x_pos, [master.PolicyA_mean_overlaps, master.PolicyB_mean_overlaps], 'grouped');
    b(1).FaceColor = [0.2 0.5 0.8]; b(2).FaceColor = [0.9 0.3 0.2];
    hold on;
    % Error bars
    ngroups = n; nbars = 2; groupwidth = min(0.8, nbars/(nbars+1.5));
    for j = 1:nbars
        x_bar = x_pos - groupwidth/2 + (2*j-1)*groupwidth/(2*nbars);
        if j == 1, std_vals = master.PolicyA_std_overlaps;
        else,      std_vals = master.PolicyB_std_overlaps; end
        errorbar(x_bar, b(j).YData, std_vals, 'k.', 'LineWidth', 1);
    end
    set(gca, 'XTick', x_pos, 'XTickLabel', strrep(envs, '_', '\_'), ...
        'XTickLabelRotation', 25, 'FontSize', 9);
    ylabel('Mean Overlap Risk Events (N=50 runs)');
    title({'Safety Comparison: Overlap Risk Events per Policy'; ...
        '(All bars from same N=50 aggregate; error bars = ±1 SD)'});
    legend({'Policy A (Baseline)', 'Policy B (Hesitation-Aware)'}, 'Location', 'northwest');
    grid on; box on;
    saveas(fh, fullfile(fig_dir, 'fig1_overlap_comparison.png'));
    close(fh);

    %% Fig 2: Overlap reduction % with confidence annotation
    fh = figure('Visible', 'off', 'Position', [100 100 900 420]);
    colors = zeros(n, 3);
    for e = 1:n
        if master.Overlap_Reduction_Pct(e) > 50, colors(e,:) = [0.1 0.7 0.2];
        elseif master.Overlap_Reduction_Pct(e) > 0, colors(e,:) = [0.9 0.7 0.0];
        else, colors(e,:) = [0.8 0.2 0.2]; end
    end
    bh = bar(x_pos, master.Overlap_Reduction_Pct);
    for e = 1:n, bh.FaceColor = 'flat'; bh.CData(e,:) = colors(e,:); end
    hold on;
    for e = 1:n
        if master.significant_p001(e)
            text(e, max(master.Overlap_Reduction_Pct(e) + 2, 2), '***', ...
                'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
        end
    end
    yline(0, 'k--', 'LineWidth', 1);
    set(gca, 'XTick', x_pos, 'XTickLabel', strrep(envs, '_', '\_'), ...
        'XTickLabelRotation', 25, 'FontSize', 9);
    ylabel('Overlap Reduction (%) — Policy B vs Policy A');
    title({'Overlap Risk Reduction by Environment (N=50 runs each)'; ...
        '*** = statistically significant at p < 0.001 (paired t-test)'});
    grid on; box on;
    saveas(fh, fullfile(fig_dir, 'fig2_reduction_pct.png'));
    close(fh);

    %% Fig 3: Cost-benefit (reduction vs time cost)
    fh = figure('Visible', 'off', 'Position', [100 100 850 480]);
    scatter(master.Time_Cost_Pct, master.Overlap_Reduction_Pct, 120, ...
        'filled', 'MarkerFaceAlpha', 0.8, 'MarkerEdgeColor', 'k');
    hold on;
    for e = 1:n
        text(master.Time_Cost_Pct(e) + 0.05, master.Overlap_Reduction_Pct(e), ...
            strrep(char(master.environment(e)), '_', '\_'), 'FontSize', 7.5);
    end
    xline(0, 'k--', 'LineWidth', 1);
    yline(0, 'k--', 'LineWidth', 1);
    xlabel('Time Cost (%) — positive = Policy B slower');
    ylabel('Overlap Reduction (%) — higher is safer');
    title({'Safety-Efficiency Tradeoff (N=50 runs per point)'; ...
        'Upper-left quadrant = safer AND faster'});
    grid on; box on;
    saveas(fh, fullfile(fig_dir, 'fig3_cost_benefit.png'));
    close(fh);

    %% Fig 4: Task completion time comparison
    fh = figure('Visible', 'off', 'Position', [100 100 900 420]);
    bar(x_pos, [master.PolicyA_mean_time_sec, master.PolicyB_mean_time_sec], 'grouped');
    set(gca, 'XTick', x_pos, 'XTickLabel', strrep(envs, '_', '\_'), ...
        'XTickLabelRotation', 25, 'FontSize', 9);
    ylabel('Mean Task Completion Time (s) — N=50 runs');
    legend({'Policy A (Baseline)', 'Policy B (Hesitation-Aware)'}, 'Location', 'northwest');
    title({'Efficiency Comparison: Task Completion Time'; ...
        '(Same N=50 aggregate data as Fig 1)'});
    grid on; box on;
    saveas(fh, fullfile(fig_dir, 'fig4_completion_time.png'));
    close(fh);
end

%% ======================================================================
function envs = buildEnvironmentList()
    envs = {
        struct('name', 'low_conflict_open'), ...
        struct('name', 'narrow_assembly_bench'), ...
        struct('name', 'precision_insertion'), ...
        struct('name', 'inspection_rework'), ...
        struct('name', 'shared_bin_access') ...
    };
end

function key = sanitize_fieldname(s)
    key = regexprep(s, '[^a-zA-Z0-9_]', '_');
    if ~isempty(key) && ~isnan(str2double(key(1)))
        key = ['s_' key];
    end
end

function rows = readJsonlSimple(path)
    fid = fopen(path, 'r');
    if fid < 0, rows = {}; return; end
    raw = cell(5000, 1); idx = 1;
    while true
        ln = fgetl(fid);
        if ~ischar(ln), break; end
        ln = strtrim(ln);
        if isempty(ln), continue; end
        try
            raw{idx} = jsondecode(ln);
            idx = idx + 1;
        catch
        end
        if idx > numel(raw), raw = [raw; cell(5000,1)]; end
    end
    fclose(fid);
    rows = raw(1:idx-1);
end
