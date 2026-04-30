% octave_compat.m  — call this once at the start of any Octave session
% Defines shims for MATLAB-only functions needed by the simulation pipeline.
% In MATLAB these are no-ops; native implementations are used.

if exist('OCTAVE_VERSION', 'builtin')
    % struct2table: convert struct array to a simple container struct
    % We don't actually use MATLAB tables in downstream logic --
    % stage3 returns struct arrays and the benchmark reads fields directly.
    % Shim returns a struct that quacks like a table for field access.
    if ~exist('struct2table', 'file')
        fid = fopen(fullfile(fileparts(mfilename('fullpath')), 'struct2table.m'), 'w');
        fprintf(fid, 'function t = struct2table(s)\n');
        fprintf(fid, '%% Octave shim: wraps a struct array for field-level access\n');
        fprintf(fid, 'if isempty(s), t = struct(); return; end\n');
        fprintf(fid, 'fn = fieldnames(s);\n');
        fprintf(fid, 't = struct();\n');
        fprintf(fid, 'for i = 1:numel(fn)\n');
        fprintf(fid, '  vals = {s.(fn{i})};\n');
        fprintf(fid, '  try\n');
        fprintf(fid, '    t.(fn{i}) = cell2mat(vals);\n');
        fprintf(fid, '  catch\n');
        fprintf(fid, '    t.(fn{i}) = vals;\n');
        fprintf(fid, '  end\n');
        fprintf(fid, 'end\n');
        fprintf(fid, 'end\n');
        fclose(fid);
    end

    if ~exist('writetable', 'file')
        fid = fopen(fullfile(fileparts(mfilename('fullpath')), 'writetable.m'), 'w');
        fprintf(fid, 'function writetable(t, path, varargin)\n');
        fprintf(fid, '%% Octave shim: writes a struct-table to CSV\n');
        fprintf(fid, 'fn = fieldnames(t);\n');
        fprintf(fid, 'fid2 = fopen(path, ''w'');\n');
        fprintf(fid, 'if fid2 < 0, warning(''writetable: cannot open %%s'', path); return; end\n');
        fprintf(fid, 'fprintf(fid2, ''%%s'', strjoin(fn, '',''));\n');
        fprintf(fid, 'fprintf(fid2, ''\\n'');\n');
        fprintf(fid, 'col1 = t.(fn{1});\n');
        fprintf(fid, 'if iscell(col1), n = numel(col1); else, n = numel(col1); end\n');
        fprintf(fid, 'for r = 1:n\n');
        fprintf(fid, '  parts = {};\n');
        fprintf(fid, '  for c = 1:numel(fn)\n');
        fprintf(fid, '    v = t.(fn{c});\n');
        fprintf(fid, '    if iscell(v), val = v{r}; else, val = v(r); end\n');
        fprintf(fid, '    if ischar(val) || islogical(val)\n');
        fprintf(fid, '      parts{end+1} = num2str(val);\n');
        fprintf(fid, '    elseif isnumeric(val)\n');
        fprintf(fid, '      parts{end+1} = num2str(val, ''%%g'');\n');
        fprintf(fid, '    else\n');
        fprintf(fid, '      parts{end+1} = char(val);\n');
        fprintf(fid, '    end\n');
        fprintf(fid, '  end\n');
        fprintf(fid, '  fprintf(fid2, ''%%s\\n'', strjoin(parts, '',''));\n');
        fprintf(fid, 'end\n');
        fprintf(fid, 'fclose(fid2);\n');
        fprintf(fid, 'end\n');
        fclose(fid);
    end

    if ~exist('table2struct', 'file')
        fid = fopen(fullfile(fileparts(mfilename('fullpath')), 'table2struct.m'), 'w');
        fprintf(fid, 'function s = table2struct(t)\n');
        fprintf(fid, '%% Octave shim: struct-table already is a struct\n');
        fprintf(fid, 's = t;\n');
        fprintf(fid, 'end\n');
        fclose(fid);
    end

    fprintf('[octave_compat] shims loaded: struct2table, writetable, table2struct\n');
end
