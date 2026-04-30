function t = struct2table(s)
% struct2table  Octave-compatible shim.
% Converts a struct array into a struct-of-arrays (column per field).
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
