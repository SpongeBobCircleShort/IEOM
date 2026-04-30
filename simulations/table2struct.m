function s = table2struct(t, varargin)
% table2struct  Dual-environment shim.
%   - MATLAB  : delegates transparently to the built-in via builtin().
%   - Octave  : provides a hand-written implementation (Octave lacks this
%               built-in when working with struct-of-columns tables).
  if ~exist('OCTAVE_VERSION', 'builtin')
    % MATLAB path: hand off to the real built-in unconditionally.
    s = builtin('table2struct', t, varargin{:});
    return;
  end
  % Octave path: hand-written struct-of-columns -> struct array conversion.
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
