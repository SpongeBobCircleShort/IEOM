function s = table2struct(t, varargin)
% table2struct  Octave-compatible shim.
% Converts a struct-of-columns back into a struct array.
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
