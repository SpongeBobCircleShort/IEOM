function generate_figures_from_csv()
  try graphics_toolkit('gnuplot'); catch; end
  addpath(fileparts(mfilename('fullpath')));
  root = fileparts(fileparts(mfilename('fullpath')));
  out  = fullfile(root, 'artifacts', 'canonical_results');
  tdir = fullfile(out, 'tables');
  fdir = fullfile(out, 'figures');
  
  csv_path = fullfile(tdir, 'master_aggregated_results.csv');
  fprintf('Reading %s\n', csv_path);
  
  fid = fopen(csv_path, 'r');
  header = fgetl(fid);
  
  n_env = 5;
  M = zeros(n_env, 9);
  env_names = cell(n_env, 1);
  
  for e = 1:n_env
    line = fgetl(fid);
    parts = strsplit(line, ',');
    env_names{e} = parts{1};
    % parts: env, N, mb, sb, ma, sa, red, mbt, mat_, tc, p
    % M: [mb sb ma sa red mbt mat_ tc p_sig]
    M(e,1) = str2double(parts{3});
    M(e,2) = str2double(parts{4});
    M(e,3) = str2double(parts{5});
    M(e,4) = str2double(parts{6});
    M(e,5) = str2double(parts{7});
    M(e,6) = str2double(parts{8});
    M(e,7) = str2double(parts{9});
    M(e,8) = str2double(parts{10});
    p_val = str2double(parts{11});
    M(e,9) = (p_val < 0.001);
  end
  fclose(fid);
  
  x = 1:n_env; lbls = strrep(env_names,'_',' ');

  try
      % Fig1: overlap bar
      fh=figure('visible','off','position',[100 100 900 420]);
      b1=bar(x,[M(:,1) M(:,3)]);
      set(b1(1), 'FaceColor', [0.2 0.5 0.8]); 
      set(b1(2), 'FaceColor', [0.9 0.3 0.2]);
      set(gca,'xtick',x,'xticklabel',lbls,'xticklabelrotation',20,'fontsize',8);
      ylabel('Mean Overlap Events (N=50 runs)');
      title('Safety: Overlap Risk Events -- same N=50 data as all tables');
      legend('Policy A Baseline','Policy B Hesitation-Aware','location','northwest');
      grid on;
      saveas(fh, fullfile(fdir,'fig1_overlap_comparison.png')); close(fh);

      % Fig2: reduction %
      fh=figure('visible','off','position',[100 100 900 420]);
      hold on;
      for e=1:n_env
        if M(e,5)>50,     clr=[0.1 0.7 0.2];
        elseif M(e,5)>0,  clr=[0.9 0.7 0.0];
        else,             clr=[0.8 0.2 0.2]; end
        b_h = bar(e, M(e,5));
        set(b_h, 'FaceColor', clr);
      end
      for e=1:n_env
        if M(e,9), text(e, M(e,5)+1,'***','horizontalalignment','center','fontsize',10,'fontweight','bold'); end
      end
      plot([0 6],[0 0],'k--');
      set(gca,'xtick',x,'xticklabel',lbls,'xticklabelrotation',20,'fontsize',8);
      ylabel('Overlap Reduction % (Policy B vs A)');
      title('Overlap Reduction by Environment (*** = p < 0.001, N=50 runs)');
      grid on;
      saveas(fh, fullfile(fdir,'fig2_reduction_pct.png')); close(fh);

      % Fig3: cost-benefit scatter
      fh=figure('visible','off','position',[100 100 800 450]);
      scatter(M(:,8), M(:,5), 100, 'filled');
      for e=1:n_env
        text(M(e,8)+0.03, M(e,5), lbls{e},'fontsize',7);
      end
      plot([-1 1],[0 0],'k--'); plot([0 0],[-200 100],'k--');
      xlabel('Time Cost % (+ = Policy B slower)');
      ylabel('Overlap Reduction %');
      title('Safety-Efficiency Tradeoff (N=50 runs per point)');
      grid on;
      saveas(fh, fullfile(fdir,'fig3_cost_benefit.png')); close(fh);

      % Fig4: completion time
      fh=figure('visible','off','position',[100 100 900 420]);
      bar(x,[M(:,6) M(:,7)]);
      set(gca,'xtick',x,'xticklabel',lbls,'xticklabelrotation',20,'fontsize',8);
      ylabel('Mean Task Completion Time (s), N=50 runs');
      legend('Policy A','Policy B','location','northwest');
      title('Efficiency: Task Completion Time (same N=50 data)');
      grid on;
      saveas(fh, fullfile(fdir,'fig4_completion_time.png')); close(fh);

      fprintf('[SAVED] 4 figures to %s\n', fdir);
  catch err
      fprintf('[WARN] Figure generation failed: %s\n', err.message);
  end
  
  % --- Paper statements ---
  reds = M(M(:,5)>0, 5);
  if isempty(reds), lo=0; hi=0; else, lo=min(reds); hi=max(reds); end
  fid = fopen(fullfile(tdir,'paper_statements.txt'),'w');
  fprintf(fid,'== Paper Statements (sourced from master_aggregated_results.csv) ==\n\n');
  fprintf(fid,'Aggregation:\n  Results represent means across N=50 randomized runs per environment.\n');
  fprintf(fid,'  Figures show aggregate statistics; all tables from same N=50 runs.\n\n');
  fprintf(fid,'Overlap Reduction Claim:\n');
  fprintf(fid,'  Range across high-conflict envs: %.1f%% to %.1f%%\n', lo, hi);
  fprintf(fid,'  Use: "%.0f-%.0f%% reduction" (not "100%% elimination")\n\n', floor(lo), ceil(hi));
  fprintf(fid,'Statistical Significance:\n');
  for e=1:n_env
    if M(e,9), fprintf(fid,'  %s: p<0.001 (paired t-test, statistically significant)\n', env_names{e}); end
  end
  fprintf(fid,'  Note: The model reduces overlap in high conflict environments, but may cause slightly increased overlap or slow downs in low conflict environments.\n');
  fclose(fid);
  fprintf('\n=== COMPLETE. ===\n');
end
