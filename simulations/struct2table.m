function t = struct2table(s)
% struct2table  Dual-environment shim.
%   - MATLAB  : delegates transparently to the built-in via builtin().
%   - Octave  : provides a hand-written implementation (Octave lacks this
%               built-in; converts a struct array to a struct-of-columns).
  if ~exist('OCTAVE_VERSION', 'builtin')
    % MATLAB path: hand off to the real built-in unconditionally.
    t = builtin('struct2table', s);
    return;
  end
  % Octave path: hand-written struct array -> struct-of-columns conversion.
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
