function out = unbiased_sa_portfolio_msv(adaptBundle, Langevin, saOpts)
%UNBIASED_SA_PORTFOLIO_MSV
% SA on xi with beta = softmax(xi), using blocked MCMC and
% truncated unbiased-style gradient estimators.
%
% Main design:
%   - At each SA iteration k:
%       * start from one common adapted state
%       * compute M replicate gradient contributions
%       * each replicate samples level l in {0,...,Lmax}
%       * B_inner = B0 * 2^l
%       * if l=0:
%             Delta_0 = mean(g^(1:B0))
%         else
%             Delta_l = (1/B_l) sum_{b=1}^{B_l/2} (g^(b+B_l/2)-g^(b))
%       * contribution = Delta_l / p_l
%       * average the M contributions
%       * update xi
%       * propagate the terminal state of replicate M
%
% Input adaptBundle must contain:
%   .model
%   .PropDist
% and optionally:
%   .accRatesAdapt
%
% Inputs:
%   adaptBundle
%   Langevin
%   saOpts must contain:
%       .Ksa
%       .gamma
%       .blockSize
%       .a0
%       .aPow
%       .xiClip
%       .seed
%       .B0
%       .Lmax
%       .level_probs      % length Lmax+1, sums to 1
%       .M                % number of replicate contributions per SA step
%       .localHorizon
%       .nForecastPerInner
%
% Optional fields:
%       .xi0
%       .verbose
%       .threadCount
%
% Output:
%   out: traces and diagnostics

    % ---------- SA / unbiased options ----------
    Ksa               = saOpts.Ksa;
    gamma             = saOpts.gamma;
    blockSize         = saOpts.blockSize;
    a0                = saOpts.a0;
    aPow              = saOpts.aPow;
    xiClip            = saOpts.xiClip;
    seed              = saOpts.seed;

    B0                = saOpts.B0;
    Lmax              = saOpts.Lmax;
    level_probs       = saOpts.level_probs(:);
    Mrep              = saOpts.M;
    localHorizon      = saOpts.localHorizon;
    nForecastPerInner = saOpts.nForecastPerInner;

    assert(length(level_probs) == Lmax + 1, 'level_probs must have length Lmax+1');
    assert(abs(sum(level_probs) - 1) < 1e-12, 'level_probs must sum to 1');

    level_cdf = cumsum(level_probs);

    verbose = true;
    if isfield(saOpts,'verbose')
        verbose = logical(saOpts.verbose);
    end

    threadCount = [];
    if isfield(saOpts,'threadCount')
        threadCount = saOpts.threadCount;
    end

    if ~isempty(threadCount)
        try
            maxNumCompThreads(threadCount);
        catch
        end
    end

    % ---------- read adapted state ----------
    if isempty(adaptBundle) || ~isstruct(adaptBundle)
        error('First input must be a non-empty adaptBundle struct.');
    end

    if ~isfield(adaptBundle,'model') || ~isfield(adaptBundle,'PropDist')
        error('adaptBundle must contain fields: .model and .PropDist');
    end

    model = adaptBundle.model;
    PropDist = adaptBundle.PropDist;

    accRatesAdapt = [];
    if isfield(adaptBundle,'accRatesAdapt')
        accRatesAdapt = adaptBundle.accRatesAdapt;
    end

    % ---------- init xi, beta ----------
    N = model.N;
    if isfield(saOpts,'xi0') && ~isempty(saOpts.xi0)
        xi = saOpts.xi0(:);
    else
        xi = zeros(N,1);
    end
    xi   = clip_vec(xi, xiClip);
    beta = softmax_stable(xi);

    % ---------- RNG stream ----------
    sMain = RandStream('Threefry','Seed',seed);

    % ---------- preallocate outputs ----------
    out.xi_hist            = zeros(N, Ksa);
    out.beta_hist          = zeros(N, Ksa);
    out.grad_xi_hist       = zeros(N, Ksa);
    out.grad_beta_hist     = zeros(N, Ksa);
    out.obj_proxy_hist     = zeros(1, Ksa);

    out.mu_hist            = zeros(N, Ksa);
    out.sigmarbeta_hist    = zeros(N, Ksa);

    out.level_draws        = zeros(Mrep, Ksa);
    out.B_draws            = zeros(Mrep, Ksa);
    out.rep_grad_norms     = zeros(Mrep, Ksa);

    out.accRates_train_lastrep = cell(1, Ksa);
    out.rngState           = cell(1, Ksa);

    out.PropDist           = PropDist;
    out.accRates_adapt     = accRatesAdapt;
    out.saOpts             = saOpts;

    % ---------- SA loop ----------
    for k = 1:Ksa

        % delayed stepsize
        ak = a0 / ((k + 100)^aPow);

        % common starting state for all M replicates at this SA step
        model_start = model;
        PropDist_start = PropDist;

        % storage for replicate contributions
        Hhat_by_rep = zeros(N, Mrep);
        Hhat_xi_by_rep = zeros(N, Mrep);

        % storage for summary quantities across replicates
        mu_rep = zeros(N, Mrep);
        sigmarbeta_rep = zeros(N, Mrep);

        % only keep the terminal state of the LAST replicate
        model_terminal_last = [];
        accRates_lastrep = [];

        % ------------------------------------------------------
        % M replicate contributions
        % ------------------------------------------------------
        for m = 1:Mrep

            % reset to the common starting state
            model_m = model_start;
            PropDist_m = PropDist_start;

            % independent substream per (k,m)
            RandStream.setGlobalStream(sMain);
            sMain.Substream = (k-1)*Mrep + m + 1;

            % sample level l in {0,...,Lmax}
            u = rand;
            l = find(u <= level_cdf, 1, 'first') - 1;
            if isempty(l)
                l = Lmax;
            end

            B_inner = B0 * 2^l;

            out.level_draws(m,k) = l;
            out.B_draws(m,k) = B_inner;

            trainOps_blk = struct();
            trainOps_blk.Burnin = 0;

            % previous/frozen outer sample for forecasting
            outer_prev = struct();
            outer_prev.h_0 = model_m.h_0;
            outer_prev.delta_0 = model_m.delta_0;
            outer_prev.sigma2_h = model_m.sigma2_h;
            outer_prev.sigma2_delta = model_m.sigma2_delta;
            outer_prev.phi_h = model_m.phi_h;
            outer_prev.phi_delta = model_m.phi_delta;

            % blocked MCMC run
            [model_m, samples_blk, accRates_blk] = mcmcTrain_blocks( ...
                model_m, PropDist_m, trainOps_blk, Langevin, B_inner);

            % forecast from all stored inner samples using frozen outer sample
            fc_blk = msv_forecast_states_only_by_blocks( ...
                samples_blk, model_m, outer_prev, localHorizon, nForecastPerInner);

            % gradients by inner sample
            [grad_by_inner, grad_xi_by_inner, ~, mu_by_inner, SigmarBeta_by_inner] = ...
                grad_msv_exact_by_blocks(beta, fc_blk, model_m, gamma, blockSize);

            % build Delta_l
            if l == 0
                Delta = mean(grad_by_inner, 2);
                Delta_xi = mean(grad_xi_by_inner, 2);
            else
                Bh = B_inner / 2;
                assert(mod(B_inner,2)==0, 'B_inner must be even for l>=1');

                Delta = sum(grad_by_inner(:,Bh+1:B_inner) - grad_by_inner(:,1:Bh), 2) / B_inner;
                Delta_xi = sum(grad_xi_by_inner(:,Bh+1:B_inner) - grad_xi_by_inner(:,1:Bh), 2) / B_inner;
            end

            % scale by 1 / p_l
            p_l = level_probs(l+1);
            Hhat_by_rep(:,m) = Delta / p_l;
            Hhat_xi_by_rep(:,m) = Delta_xi / p_l;

            % summary quantities for proxy diagnostics
            mu_rep(:,m) = mean(mu_by_inner, 2);
            sigmarbeta_rep(:,m) = mean(SigmarBeta_by_inner, 2);

            out.rep_grad_norms(m,k) = norm(Hhat_xi_by_rep(:,m));

            % keep terminal state of the last replicate only
            if m == Mrep
                model_terminal_last = model_m;
                accRates_lastrep = accRates_blk;
            end
        end

        % ------------------------------------------------------
        % Average replicate contributions
        % ------------------------------------------------------
        Hhat_mean = mean(Hhat_by_rep, 2);
        Hhat_xi_mean = mean(Hhat_xi_by_rep, 2);

        mu_mean = mean(mu_rep, 2);
        SigmarBeta_mean = mean(sigmarbeta_rep, 2);

        % proxy at current beta before update
        obj_proxy_old = beta' * mu_mean(:) - (gamma/2) * (beta' * SigmarBeta_mean(:));

        % ------------------------------------------------------
        % SA update on xi
        % ------------------------------------------------------
        xi = xi + ak * Hhat_xi_mean;
        xi = clip_vec(xi, xiClip);
        beta = softmax_stable(xi);

        % ------------------------------------------------------
        % Propagate terminal chain state from last replicate
        % ------------------------------------------------------
        model = model_terminal_last;

        % ------------------------------------------------------
        % Store
        % ------------------------------------------------------
        out.xi_hist(:,k)         = xi;
        out.beta_hist(:,k)       = beta;
        out.grad_xi_hist(:,k)    = Hhat_xi_mean;
        out.grad_beta_hist(:,k)  = Hhat_mean;
        out.mu_hist(:,k)         = mu_mean(:);
        out.sigmarbeta_hist(:,k) = SigmarBeta_mean(:);
        out.obj_proxy_hist(k)    = obj_proxy_old;

        out.accRates_train_lastrep{k} = accRates_lastrep;
        out.rngState{k} = struct('main', sMain.State);

        if verbose
            fprintf('USA iter %d | a_k=%.3g | obj_proxy=%.6g | mean level=%.3f\n', ...
                k, ak, out.obj_proxy_hist(k), mean(out.level_draws(:,k)));
        end
    end

    out.xi_final   = xi;
    out.beta_final = beta;
end

% ===== helpers =====
function beta = softmax_stable(xi)
    z = xi - max(xi);
    e = exp(z);
    beta = e ./ sum(e);
end

function x = clip_vec(x, clipVal)
    if isempty(clipVal) || ~isfinite(clipVal)
        return;
    end
    x = max(min(x, clipVal), -clipVal);
end