function t = struct2table(s)
% struct2table  Octave-compatible shim. In MATLAB the native version is used.
% Converts a struct array into a struct-of-arrays (column per field).
% Downstream code accesses t.field_name which works the same way.
  if ~exist('OCTAVE_VERSION', 'builtin')
    % Let MATLAB use its own built-in — this file should never be on
    % MATLAB's path ahead of the built-in, but guard anyway.
    error('struct2table shim called in MATLAB — check your path order.');
  end
  if isempty(s)
    t = struct();
    return;
  end
  fn = fieldnames(s);
  t  = struct();
  for i = 1:numel(fn)
    f = fn{i};
    vals = {s.(f)};
    % Try to flatten to numeric array; fall back to cell
    try
      flat = cell2mat(vals);
      if numel(flat) == numel(s)
        t.(f) = flat(:);
      else
        t.(f) = vals(:);
      end
    catch
      t.(f) = vals(:);
    end
  end
end
