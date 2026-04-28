function [out, adaptBundle] = unbiased_sa_portfolio_msv_online(adaptBundle, mcmcoptions, Langevin, onlineOpts)
%UNBIASED_SA_PORTFOLIO_MSV_ONLINE
% Online wrapper around unbiased_sa_portfolio_msv, using a loop over
% REBALANCE DATES ONLY.
%
% Design:
%   - Rebalance every rebalanceEvery dates
%   - At each rebalance date, build the day model using data up to that date
%   - Call unbiased_sa_portfolio_msv once
%   - Warm start both:
%       * xi
%       * chain/model state via adaptBundle.model
%   - Keep adaptBundle.PropDist fixed (do not rerun mcmcAdapt)
%   - Fill daily beta/xi outputs by holding them constant between rebalances
%   - The local engine is assumed to propagate model_final using the
%     replicate with the LARGEST sampled level
%
% Inputs:
%   adaptBundle : struct with fields
%       .model
%       .PropDist
%       .accRatesAdapt   (optional)
%
%   mcmcoptions : kept for interface compatibility / future use
%   Langevin
%   onlineOpts : struct with fields
%       .T0               : initial training length
%       .H                : number of online calendar dates to output
%       .saInnerIters     : SA iterations per rebalance call
%       .saOptsBase       : base options for unbiased_sa_portfolio_msv
%
% Optional onlineOpts fields:
%       .xi0                      : initial xi
%       .forceModelHorizon        : logical, default false
%       .modelHorizonValue        : if forceModelHorizon=true, set model.horizon
%       .reseedPerRebalance       : logical, default true
%       .rebalanceEvery           : default saOptsBase.localHorizon
%       .verbose                  : logical, default true
%
% Outputs:
%   out.beta(:,h)        : portfolio weights for each calendar date h=1,...,H
%   out.xi(:,h)          : xi for each calendar date
%   out.obj_proxy(h)     : objective proxy, constant between rebalances
%   out.rebalanceFlag(h) : true on rebalance dates, false otherwise
%   out.day{h}           : outDay struct on rebalance dates, [] otherwise
%   out.kOffset(h)       : global SA offset after date h
%   out.rebalanceDates   : actual rebalance calendar indices (relative to online horizon)
%   out.selectedLevel(h) : selected propagated level on rebalance dates, NaN otherwise

    if nargin < 4
        error('Usage: [out, adaptBundle] = unbiased_sa_portfolio_msv_online(adaptBundle, mcmcoptions, Langevin, onlineOpts)');
    end

    if isempty(adaptBundle) || ~isstruct(adaptBundle)
        error('adaptBundle must be a non-empty struct.');
    end
    if ~isfield(adaptBundle,'model') || ~isfield(adaptBundle,'PropDist')
        error('adaptBundle must contain fields .model and .PropDist');
    end

    modelFull = adaptBundle.model;

    if ~isfield(modelFull,'Y') || isempty(modelFull.Y)
        error('adaptBundle.model must contain field Y with the full data panel.');
    end

    Yfull = modelFull.Y;
    [N, TT] = size(Yfull);

    % ------------------------------------------------------------
    % Parse online options
    % ------------------------------------------------------------
    if ~isfield(onlineOpts,'T0') || ~isfield(onlineOpts,'H')
        error('onlineOpts must contain fields T0 and H.');
    end

    T0 = onlineOpts.T0;
    H  = onlineOpts.H;

    if TT < T0 + H - 1
        error('adaptBundle.model.Y must have at least T0+H-1 columns.');
    end

    if ~isfield(onlineOpts,'saInnerIters') || isempty(onlineOpts.saInnerIters)
        onlineOpts.saInnerIters = 1;
    end
    saInnerIters = onlineOpts.saInnerIters;

    if ~isfield(onlineOpts,'saOptsBase') || isempty(onlineOpts.saOptsBase)
        error('onlineOpts.saOptsBase must be provided.');
    end
    saOptsBase = onlineOpts.saOptsBase;

    verbose = true;
    if isfield(onlineOpts,'verbose') && ~isempty(onlineOpts.verbose)
        verbose = logical(onlineOpts.verbose);
    end

    reseedPerRebalance = true;
    if isfield(onlineOpts,'reseedPerRebalance') && ~isempty(onlineOpts.reseedPerRebalance)
        reseedPerRebalance = logical(onlineOpts.reseedPerRebalance);
    end

    if isfield(onlineOpts,'rebalanceEvery') && ~isempty(onlineOpts.rebalanceEvery)
        rebalanceEvery = onlineOpts.rebalanceEvery;
    else
        if ~isfield(saOptsBase,'localHorizon') || isempty(saOptsBase.localHorizon)
            error('Either onlineOpts.rebalanceEvery or saOptsBase.localHorizon must be provided.');
        end
        rebalanceEvery = saOptsBase.localHorizon;
    end

    forceModelHorizon = false;
    if isfield(onlineOpts,'forceModelHorizon') && ~isempty(onlineOpts.forceModelHorizon)
        forceModelHorizon = logical(onlineOpts.forceModelHorizon);
    end

    modelHorizonValue = [];
    if isfield(onlineOpts,'modelHorizonValue') && ~isempty(onlineOpts.modelHorizonValue)
        modelHorizonValue = onlineOpts.modelHorizonValue;
    end

    % ------------------------------------------------------------
    % Initialize xi and beta
    % ------------------------------------------------------------
    if isfield(onlineOpts,'xi0') && ~isempty(onlineOpts.xi0)
        xi = onlineOpts.xi0(:);
    elseif isfield(saOptsBase,'xi0') && ~isempty(saOptsBase.xi0)
        xi = saOptsBase.xi0(:);
    else
        xi = zeros(N,1);
    end
    beta = softmax_stable(xi);

    % ------------------------------------------------------------
    % Allocate daily outputs
    % ------------------------------------------------------------
    out.beta          = zeros(N, H);
    out.xi            = zeros(N, H);
    out.obj_proxy     = NaN(1, H);
    out.rebalanceFlag = false(1, H);
    out.day           = cell(1, H);
    out.kOffset       = zeros(1, H);

    out.rebalanceEvery = rebalanceEvery;
    out.localHorizon   = saOptsBase.localHorizon;
    out.rebalanceDates = 1:rebalanceEvery:H;
    out.selectedLevel  = NaN(1, H);

    % global SA counter for step-size continuity
    kGlob = 0;

    % ------------------------------------------------------------
    % Loop over REBALANCE DATES ONLY
    % ------------------------------------------------------------
    rebalanceDates = out.rebalanceDates;
    nReb = numel(rebalanceDates);

    for r = 1:nReb
        hStart = rebalanceDates(r);
        hEnd   = min(hStart + rebalanceEvery - 1, H);

        Tcur = T0 + hStart - 1;

        % --------------------------------------------------------
        % Build rebalance-day model from full data
        % --------------------------------------------------------
        modelDay = modelFull;
        modelDay.Y = Yfull(:, 1:Tcur);
        modelDay.N = N;
        modelDay.T = Tcur;

                % --------------------------------------------------------
        % Resize time-varying fields to match Tcur
        % --------------------------------------------------------
        tvFields = {'Ft','hs','deltas','omegas','lambdas'};
        for ii = 1:numel(tvFields)
            f = tvFields{ii};
            if isfield(modelDay,f) && ~isempty(modelDay.(f))
                A = modelDay.(f);
                if size(A,2) >= Tcur
                    modelDay.(f) = A(:,1:Tcur);
                else
                    modelDay.(f) = [A, repmat(A(:,end),1,Tcur-size(A,2))];
                end
            end
        end

        if isfield(modelDay,'deltaFactors') && ~isempty(modelDay.deltaFactors)
            dF = modelDay.deltaFactors(:);
            if numel(dF) >= Tcur
                modelDay.deltaFactors = dF(1:Tcur);
            else
                modelDay.deltaFactors = [dF; repmat(dF(end), Tcur-numel(dF), 1)];
            end
        end
        

        if forceModelHorizon
            if isempty(modelHorizonValue)
                modelDay.horizon = rebalanceEvery;
            else
                modelDay.horizon = modelHorizonValue;
            end
        end

        % --------------------------------------------------------
        % Warm-start chain/model state
        % --------------------------------------------------------
        modelDay = transplant_model_state(modelDay, adaptBundle.model);

        % --------------------------------------------------------
        % Build day-specific SA options
        % --------------------------------------------------------
        saOptsDay = saOptsBase;
        saOptsDay.Ksa = saInnerIters;
        saOptsDay.xi0 = xi;
        saOptsDay.kOffset = kGlob;

        if reseedPerRebalance
            if ~isfield(saOptsBase,'seed') || isempty(saOptsBase.seed)
                baseSeed = 1;
            else
                baseSeed = saOptsBase.seed;
            end
            saOptsDay.seed = baseSeed + 1000*r;
        end

        % Update the bundle's model to the current rebalance-day model
        adaptBundle.model = modelDay;

        % --------------------------------------------------------
        % Local unbiased SA call
        % --------------------------------------------------------
        outDay = unbiased_sa_portfolio_msv(adaptBundle, Langevin, saOptsDay);

        % --------------------------------------------------------
        % Warm-start xi and beta for next rebalance
        % --------------------------------------------------------
        if isfield(outDay,'xi_final') && ~isempty(outDay.xi_final)
            xi = outDay.xi_final(:);
        else
            error('unbiased_sa_portfolio_msv output must contain out.xi_final');
        end

        if isfield(outDay,'beta_final') && ~isempty(outDay.beta_final)
            beta = outDay.beta_final(:);
        else
            beta = softmax_stable(xi);
        end

        % --------------------------------------------------------
        % Warm-start chain/model state for next rebalance
        % Assumes outDay.model_final comes from the replicate with largest level
        % --------------------------------------------------------
        if isfield(outDay,'model_final') && ~isempty(outDay.model_final)
            adaptBundle.model = outDay.model_final;
        else
            adaptBundle.model = modelDay;
        end

        % --------------------------------------------------------
        % Store REBALANCE-DAY output
        % --------------------------------------------------------
        out.rebalanceFlag(hStart) = true;
        out.day{hStart} = outDay;

        if isfield(outDay,'obj_proxy_hist') && ~isempty(outDay.obj_proxy_hist)
            objVal = outDay.obj_proxy_hist(end);
        elseif isfield(outDay,'obj_proxy_final') && ~isempty(outDay.obj_proxy_final)
            objVal = outDay.obj_proxy_final;
        else
            objVal = NaN;
        end

        if isfield(outDay,'selectedReplicateLevel') && ~isempty(outDay.selectedReplicateLevel)
            out.selectedLevel(hStart) = outDay.selectedReplicateLevel;
        end

        % advance global SA counter
        kGlob = kGlob + saInnerIters;

        % --------------------------------------------------------
        % Fill DAILY outputs over the holding interval
        % --------------------------------------------------------
        out.beta(:, hStart:hEnd) = repmat(beta, 1, hEnd-hStart+1);
        out.xi(:, hStart:hEnd)   = repmat(xi,   1, hEnd-hStart+1);
        out.obj_proxy(hStart:hEnd) = objVal;
        out.kOffset(hStart:hEnd) = kGlob;

        if verbose
            if isfield(outDay,'selectedReplicateLevel') && ~isempty(outDay.selectedReplicateLevel)
                fprintf('[unbiased-online] rebalance %d/%d | online day %d | T=%d | hold [%d,%d] | obj_proxy=%g | propagated level=%d\n', ...
                    r, nReb, hStart, Tcur, hStart, hEnd, objVal, outDay.selectedReplicateLevel);
            else
                fprintf('[unbiased-online] rebalance %d/%d | online day %d | T=%d | hold [%d,%d] | obj_proxy=%g\n', ...
                    r, nReb, hStart, Tcur, hStart, hEnd, objVal);
            end
        end
    end

    % ------------------------------------------------------------
    % Fill non-rebalance day cells with []
    % ------------------------------------------------------------
    nonReb = setdiff(1:H, rebalanceDates);
    for j = 1:numel(nonReb)
        out.day{nonReb(j)} = [];
    end
end

% ============================================================
% Helpers
% ============================================================

function beta = softmax_stable(xi)
    xi = xi(:);
    z = xi - max(xi);
    ez = exp(z);
    beta = ez / sum(ez);
end

function modelNew = transplant_model_state(modelNew, modelOld)
% Best-effort transfer of current chain/model state into the rebalance-day model.

    if isempty(modelOld) || ~isstruct(modelOld)
        return;
    end

    copyFields = { ...
        'sigma2', 'h_0', 'delta_0', ...
        'sigma2_h', 'sigma2_delta', ...
        'phi_h', 'phi_delta', ...
        'tildephi_h', 'tildephi_delta', ...
        'auxLikVar', 'deltaFactors', ...
        'Weights' ...
    };

    for i = 1:numel(copyFields)
        f = copyFields{i};
        if isfield(modelOld, f)
            modelNew.(f) = modelOld.(f);
        end
    end

    tvFields = {'Ft','hs','deltas','omegas'};

    for i = 1:numel(tvFields)
        f = tvFields{i};
        if isfield(modelNew,f) && isfield(modelOld,f) ...
                && ~isempty(modelNew.(f)) && ~isempty(modelOld.(f)) ...
                && ndims(modelNew.(f)) == 2 && ndims(modelOld.(f)) == 2

            Anew = modelNew.(f);
            Aold = modelOld.(f);

            if size(Anew,1) == size(Aold,1)
                Tnew = size(Anew,2);
                Told = size(Aold,2);

                Tmin = min(Tnew, Told);
                Anew(:,1:Tmin) = Aold(:,1:Tmin);

                if Tnew > Told
                    Anew(:,Told+1:Tnew) = repmat(Aold(:,Told), 1, Tnew-Told);
                end

                modelNew.(f) = Anew;
            end
        end
    end
end