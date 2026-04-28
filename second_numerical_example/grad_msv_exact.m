function [grad_mean,grad_xi_mean, grad_hs, mu_mean, SigmarBeta_mean] = grad_msv_exact( ...
    beta, fc_only, model, gamma, blockSize)
%GRAD_MSV_EXACT  Mean gradient over horizon and posterior draws.
%
% Objective per (h,s):
%   f_{h,s}(beta) = beta' * mu_r(h,s) - (gamma/2) * beta' * Sigmar(h,s) * beta
% Gradient per (h,s):
%   g_{h,s}(beta) = mu_r(h,s) - gamma * Sigmar(h,s)*beta
%
% Inputs:
%   beta      : N x 1 portfolio weights
%   fc_only   : output of msv_forecast_states_only(..., 'staticmean' or 'persample')
%               must contain lambdas_fc (K x H x S), omegas_fc (R x H x S)
%               and static.LW_mean (N x K), static.sigma2_mean (scalar or N-vector)
%   model     : must contain N,K,horizon,Givset
%   gamma     : risk aversion scalar
%   blockSize : passed to Sigmar_times_beta_exact
%
% Outputs:
%   grad_mean       : N x 1 mean gradient over (h,s)
%   grad_hs         : N x H x S gradients per (h,s) (optional)
%   mu_mean         : N x 1 mean mu_r over (h,s) (optional)
%   SigmarBeta_mean : N x 1 mean Sigmar*beta over (h,s) (optional)

    if nargin < 5 || isempty(blockSize)
        blockSize = 512;
    end

    N = model.N;
    H = model.horizon;
    S = size(fc_only.lambdas_fc, 3);

    % Static params (staticmean case). If you want perSample, you need LW/sigma2 per s.
    LW   = fc_only.static.LW_mean;        % N x K
    sig2 = fc_only.static.sigma2_mean;    % scalar or N-vector

    grad_hs = zeros(N, H, S);
    mu_acc  = zeros(N, 1);
    sb_acc  = zeros(N, 1);

    % Loop over posterior draws and horizons
    for s = 1:S
        for hh = 1:H
            omegas  = fc_only.omegas_fc(:, hh, s);
            lambdas = fc_only.lambdas_fc(:, hh, s);

            % Exact action + mean for this (h,s)
            [Sigmar_beta, mu_r] = Sigmar_times_beta_exact( ...
                beta, LW, omegas, lambdas, sig2, model.Givset, blockSize);

            Sigmar_beta = Sigmar_beta(:);
            mu_r        = mu_r(:);

            g = mu_r - gamma * Sigmar_beta;

            grad_hs(:, hh, s) = g;
            mu_acc = mu_acc + mu_r;
            sb_acc = sb_acc + Sigmar_beta;
        end
    end

    denom = H * S;
    grad_mean       = sum(grad_hs, [2 3]) / denom;
    mu_mean         = mu_acc / denom;
    SigmarBeta_mean = sb_acc / denom;


    grad_xi_mean=-beta.*(beta'*grad_mean)+beta.*grad_mean;
end