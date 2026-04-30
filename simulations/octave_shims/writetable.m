function writetable(t, path, varargin)
% writetable  Octave-compatible shim. Writes a table/struct-of-columns to CSV.
  % Octave path: hand-written struct-of-columns -> CSV writer.
  fn = fieldnames(t);
  if isempty(fn)
    fid = fopen(path, 'w'); if fid >= 0, fclose(fid); end
    return;
  end
  fid = fopen(path, 'w');
  if fid < 0
    warning('writetable: cannot open %s for writing', path);
    return;
  end
  % Header
  fprintf(fid, '%s\n', strjoin(fn, ','));
  % Determine number of rows
  col1 = t.(fn{1});
  if iscell(col1), n = numel(col1); else, n = numel(col1); end
  % Rows
  for r = 1:n
    parts = cell(1, numel(fn));
    for c = 1:numel(fn)
      v = t.(fn{c});
      if iscell(v), val = v{r}; else, val = v(r); end
      if ischar(val)
        parts{c} = val;
      elseif islogical(val) || (isnumeric(val) && isscalar(val))
        parts{c} = num2str(double(val), '%g');
      else
        parts{c} = char(val);
      end
    end
    fprintf(fid, '%s\n', strjoin(parts, ','));
  end
  fclose(fid);
end
