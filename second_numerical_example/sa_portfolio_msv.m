function [out,adaptBundle] = sa_portfolio_msv(model, mcmcoptions, Langevin, saOpts, adaptBundle)
%SA_PORTFOLIO_MSV  SA on xi with beta = softmax(xi), using MSV inner loop.
%
% If PropDist_in is provided and non-empty, adaptation is skipped and that
% proposal is used. Otherwise, mcmcAdapt is run once.
    % ---------- SA options ----------
    Ksa       = saOpts.Ksa;
    gamma     = saOpts.gamma;
    blockSize = saOpts.blockSize;
    a0        = saOpts.a0;
    aPow      = saOpts.aPow;
    xiClip    = saOpts.xiClip;
    seed      = saOpts.seed;

    verbose = true;
    if isfield(saOpts,'verbose'), verbose = logical(saOpts.verbose); end

    threadCount = [];
    if isfield(saOpts,'threadCount'), threadCount = saOpts.threadCount; end

    % ---------- deterministic threading (optional) ----------
    if ~isempty(threadCount)
        try, maxNumCompThreads(threadCount); catch, end
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

    % ---------- RNG streams ----------
    sMCMC = RandStream('Threefry','Seed',seed);
    sFC   = RandStream('Threefry','Seed',seed+1);

    % ---------- (optional) adapt once ----------
    
    % --- Adaptation bundle handling ---
    if nargin < 5 || isempty(adaptBundle)
        % run adaptation once
        RandStream.setGlobalStream(sMCMC);
        sMCMC.Substream = 1;

        [model, PropDist, ~, accRatesAdapt] = mcmcAdapt(model, mcmcoptions.adapt, Langevin);

        adaptBundle = struct();
        adaptBundle.model = model;         % IMPORTANT: adapted model (has auxLikVar etc.)
        adaptBundle.PropDist = PropDist;
        adaptBundle.accRatesAdapt = accRatesAdapt;
        didAdapt = true;
    else
        % reuse previous adaptation
        if ~isfield(adaptBundle,'model') || ~isfield(adaptBundle,'PropDist')
            error('adaptBundle must contain fields: .model and .PropDist');
        end
        model = adaptBundle.model;         % IMPORTANT: bring over auxLikVar etc.
        PropDist = adaptBundle.PropDist;
        didAdapt = false;
        accRatesAdapt = [];
        if isfield(adaptBundle,'accRatesAdapt'), accRatesAdapt = adaptBundle.accRatesAdapt; end
    end


    % ---------- preallocate outputs ----------
    out.xi_hist          = zeros(N, Ksa);
    out.beta_hist        = zeros(N, Ksa);
    out.grad_xi_hist     = zeros(N, Ksa);
    out.grad_beta_hist   = zeros(N, Ksa);
    out.mu_hist          = zeros(N, Ksa);
    out.sigmarbeta_hist  = zeros(N, Ksa);
    out.obj_proxy_hist   = zeros(1, Ksa);
    out.accRates_train   = cell(1, Ksa);
    out.rngState         = cell(1, Ksa);

    out.PropDist       = PropDist;
    out.didAdapt       = didAdapt;
    out.accRates_adapt = accRatesAdapt;
    out.saOpts         = saOpts;

    % ---------- SA loop ----------
    for k = 1:Ksa

        % delayed stepsize
        ak = a0 / ((k + 100)^aPow);

        % (1) MCMC train (deterministic per k)
        RandStream.setGlobalStream(sMCMC);
        sMCMC.Substream = k;
        [model, samples, accRatesTrain] = mcmcTrain(model, PropDist, mcmcoptions.train, Langevin);

        % (2) Forecast states only (deterministic per k)
        RandStream.setGlobalStream(sFC);
        sFC.Substream = k;
        fc_only = msv_forecast_states_only(samples, model, 'staticmean');

        % (3) Gradient
        [grad_beta_mean, grad_xi_mean, ~, mu_mean, SigmarBeta_mean] = ...
            grad_msv_exact(beta, fc_only, model, gamma, blockSize);

        % --- evaluate proxy at the SAME beta used above ---
        obj_proxy_old = beta' * mu_mean(:) - (gamma/2) * (beta' * SigmarBeta_mean(:));


        grad_beta_mean = grad_beta_mean(:);
        grad_xi_mean   = grad_xi_mean(:);

        % (4) SA update on xi (maximize)
        xi = xi + ak * grad_xi_mean;
        xi = clip_vec(xi, xiClip);
        beta = softmax_stable(xi);
        % (5) store
        out.xi_hist(:,k)        = xi;
        out.beta_hist(:,k)      = beta;
        out.grad_xi_hist(:,k)   = grad_xi_mean;
        out.grad_beta_hist(:,k) = grad_beta_mean;
        out.mu_hist(:,k)        = mu_mean(:);
        out.sigmarbeta_hist(:,k)= SigmarBeta_mean(:);

        %out.obj_proxy_hist(k) = beta' * mu_mean(:) - (gamma/2) * (beta' * SigmarBeta_mean(:));
        out.obj_proxy_hist(k) = obj_proxy_old;
        out.accRates_train{k} = accRatesTrain;

        out.rngState{k} = struct('mcmc', sMCMC.State, 'fc', sFC.State);

        if verbose
            fprintf('SA iter %d | a_k=%.3g | obj_proxy=%.6g\n', k, ak, out.obj_proxy_hist(k));
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
    if isempty(clipVal) || ~isfinite(clipVal), return; end
    x = max(min(x, clipVal), -clipVal);
end