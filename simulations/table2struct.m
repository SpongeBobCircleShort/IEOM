function s = table2struct(t, varargin)
% table2struct  Octave-compatible shim.
% In our pipeline, struct2table produces a struct-of-columns,
% and table2struct is called in run_paper_benchmark to convert rows back.
% We do the inverse: struct-of-columns -> struct array.
  if ~exist('OCTAVE_VERSION', 'builtin')
    error('table2struct shim called in MATLAB — check your path order.');
  end
  fn = fieldnames(t);
  if isempty(fn), s = struct([]); return; end
  col1 = t.(fn{1});
  if iscell(col1), n = numel(col1); else, n = numel(col1); end
  s = struct();
  for r = 1:n
    for c = 1:numel(fn)
      v = t.(fn{c});
      if iscell(v), s(r).(fn{c}) = v{r};
      else,         s(r).(fn{c}) = v(r); end
    end
  end
end
