function summary = run_ab_scenarios(varargin)
% run_ab_scenarios: Entry point for Stage 1/2/3 A/B simulation scaffolds.
%
% Defaults to Stage 3:
%   run_ab_scenarios
% Optional:
%   run_ab_scenarios('stage', 'stage1')
%   run_ab_scenarios('stage', 'stage2', 'enable_replay', true)
%   run_ab_scenarios('stage', 'stage3', 'use_stub', false)

    repo_root = fileparts(mfilename('fullpath'));
    addpath(fullfile(repo_root, 'simulations'));

    stage = 'stage3';
    passthrough = varargin;
    if mod(numel(varargin), 2) == 0
        for idx = 1:2:numel(varargin)
            if strcmpi(char(varargin{idx}), 'stage')
                stage = lower(char(varargin{idx + 1}));
                passthrough(idx:idx+1) = [];
                break;
            end
        end
    end

    switch stage
        case 'stage1'
            summary = stage1_run_ab_scenarios();
        case 'stage2'
            summary = stage2_run_ab_scenarios(passthrough{:});
        case 'stage3'
            summary = stage3_run_ab_scenarios(passthrough{:});
        otherwise
            error('Unknown stage: %s', stage);
    end
end
