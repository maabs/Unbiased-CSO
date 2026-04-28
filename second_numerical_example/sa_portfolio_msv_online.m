function [out, adaptBundle] = sa_portfolio_msv_online(modelFull, mcmcoptions, Langevin, onlineOpts, adaptBundle)
%SA_PORTFOLIO_MSV_ONLINE  Online/expanding-window SA using sa_portfolio_msv.
%
% Assumptions:
%   1) modelFull already contains the full panel in modelFull.Y (N x (T0+H)).
%   2) sa_portfolio_msv has been updated to accept an optional 5th argument:
%        [outDay, adaptBundle] = sa_portfolio_msv(model, mcmcoptions, Langevin, saOpts, adaptBundle)
%      where adaptBundle is empty on the first call (to run mcmcAdapt) and
%      non-empty thereafter (to reuse PropDist and adapted model fields such as auxLikVar).
%
% Inputs:
%   modelFull   : struct with at least fields Y (N x TT), N (optional), T (optional),
%                 and all other MSV model fields required by mcmcAdapt/mcmcTrain.
%   mcmcoptions : struct with fields .adapt and .train as in your demos
%   Langevin    : 0/1
%   onlineOpts  : struct with fields:
%       - T0            : initial training length
%       - H             : number of online steps (days) to optimize
%       - saInnerIters  : SA iterations per day (passed to sa_portfolio_msv via saOpts.Ksa)
%       - saOptsBase    : base SA options struct (gamma, a0, aPow, blockSize, xiClip, seed, verbose, etc.)
%       - xi0           : optional initial xi (default zeros)
%       - forceHorizon1 : optional (default true) set model.horizon = 1 each day
%       - reseedPerDay  : optional (default true) makes runs reproducible across cell-by-cell execution
%
%   adaptBundle : optional; if provided non-empty, skips adaptation on first day
%
% Outputs:
%   out.beta(:,h)       : portfolio weights after day h optimization
%   out.xi(:,h)         : xi after day h optimization
%   out.obj_proxy(h)    : last objective proxy from day h
%   out.day{h}          : full outDay struct returned by sa_portfolio_msv
%   adaptBundle         : updated bundle (contains last model state + PropDist)
%
% Notes:
%   - This wrapper uses an expanding window: day h uses Y(:,1:T0+h-1).
%   - SA stepsize continuity across days is handled by saOpts.kOffset (requires
%     sa_portfolio_msv to support kOffset in its stepsize schedule).
%
% Copy-paste ready.

    if nargin < 5
        adaptBundle = [];
    end

    if ~isfield(modelFull,'Y') || isempty(modelFull.Y)
        error('modelFull must contain modelFull.Y (N x TT).');
    end

    Yfull = modelFull.Y;
    [N, TT] = size(Yfull);

    % ---- parse options ----
    if ~isfield(onlineOpts,'T0') || ~isfield(onlineOpts,'H')
        error('onlineOpts must contain fields T0 and H.');
    end
    T0 = onlineOpts.T0;
    H  = onlineOpts.H;

    if TT < T0 + H
        error('modelFull.Y must have at least T0+H columns.');
    end

    if ~isfield(onlineOpts,'saInnerIters') || isempty(onlineOpts.saInnerIters)
        onlineOpts.saInnerIters = 1;
    end
    saInnerIters = onlineOpts.saInnerIters;

    if ~isfield(onlineOpts,'saOptsBase') || isempty(onlineOpts.saOptsBase)
        error('onlineOpts.saOptsBase must be provided (base SA options).');
    end
    saOptsBase = onlineOpts.saOptsBase;

    forceHorizon1 = true;
    if isfield(onlineOpts,'forceHorizon1') && ~isempty(onlineOpts.forceHorizon1)
        forceHorizon1 = logical(onlineOpts.forceHorizon1);
    end

    reseedPerDay = true;
    if isfield(onlineOpts,'reseedPerDay') && ~isempty(onlineOpts.reseedPerDay)
        reseedPerDay = logical(onlineOpts.reseedPerDay);
    end

    if ~isfield(saOptsBase,'seed') || isempty(saOptsBase.seed)
        saOptsBase.seed = 1;
    end
    baseSeed = saOptsBase.seed;

    % ---- init xi ----
    if isfield(onlineOpts,'xi0') && ~isempty(onlineOpts.xi0)
        xi = onlineOpts.xi0(:);
    elseif isfield(saOptsBase,'xi0') && ~isempty(saOptsBase.xi0)
        xi = saOptsBase.xi0(:);
    else
        xi = zeros(N,1);
    end

    % ---- allocate outputs ----
    out.beta      = zeros(N, H);
    out.xi        = zeros(N, H);
    out.obj_proxy = zeros(1, H);
    out.day       = cell(1, H);

    % global SA counter for stepsize continuity
    kGlob = 0;

    % ---- online loop ----
    for h = 1:H

        % Expanding window length
        Tcur = T0 + (h-1);

        % Slice data into a per-day model
        model = modelFull;
        model.Y = Yfull(:, 1:Tcur);
        model.N = N;
        model.T = Tcur;

        if forceHorizon1
            model.horizon = 1;
        end

        % Day SA options
        saOptsDay = saOptsBase;
        saOptsDay.Ksa = saInnerIters;
        saOptsDay.xi0 = xi;

        % Stepsize continuity across days (requires sa_portfolio_msv to use kOffset)
        saOptsDay.kOffset = kGlob;

        % Optional deterministic reseeding per day to prevent "cell-by-cell" RNG drift
        if reseedPerDay
            saOptsDay.seed = baseSeed + 1000*h;
        end

        % Call engine
        [outDay, adaptBundle] = sa_portfolio_msv(model, mcmcoptions, Langevin, saOptsDay, adaptBundle);

        % Warm start next day
        if isfield(outDay,'xi_final')
            xi = outDay.xi_final(:);
        elseif isfield(outDay,'xi_end')
            xi = outDay.xi_end(:);
        else
            error('sa_portfolio_msv output must contain xi_final (or xi_end).');
        end

        % Extract beta
        if isfield(outDay,'beta_final')
            beta = outDay.beta_final(:);
        elseif isfield(outDay,'beta_end')
            beta = outDay.beta_end(:);
        else
            % If not provided, reconstruct from xi via softmax (stable)
            beta = softmax_stable(xi);
        end

        % Extract objective proxy
        if isfield(outDay,'obj_proxy_hist') && ~isempty(outDay.obj_proxy_hist)
            objp = outDay.obj_proxy_hist(end);
        elseif isfield(outDay,'obj_proxy_end')
            objp = outDay.obj_proxy_end;
        else
            objp = NaN;
        end

        out.beta(:,h)      = beta;
        out.xi(:,h)        = xi;
        out.obj_proxy(h)   = objp;
        out.day{h}         = outDay;

        % advance global SA iteration counter
        kGlob = kGlob + saInnerIters;

        if isfield(saOptsBase,'verbose') && saOptsBase.verbose
            fprintf('[online] day %d/%d | T=%d | obj_proxy=%g\n', h, H, Tcur, objp);
        end
    end
end


% -------------------- helper: stable softmax --------------------
function beta = softmax_stable(xi)
    xi = xi(:);
    z  = xi - max(xi);
    ez = exp(z);
    beta = ez / sum(ez);
end