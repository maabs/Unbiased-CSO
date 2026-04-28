function [grad_by_inner, grad_xi_by_inner, grad_hbs, mu_by_inner, SigmarBeta_by_inner] = ...
    grad_msv_exact_by_blocks(beta, fc_blk, model, gamma, blockSize)
%GRAD_MSV_EXACT_BY_BLOCKS
% Compute exact gradients for blocked forecasts.
%
% This version consumes the output of
% msv_forecast_states_only_by_blocks(...).
%
% For each:
%   - inner sample b = 1,...,B_inner
%   - forecast replication s = 1,...,S_fc
%   - local horizon step hh = 1,...,H_local
%
% it computes:
%   g_{hh,b,s}(beta) = mu_r(hh,b,s) - gamma * Sigmar(hh,b,s) * beta
%
% It RETURNS:
%   - the full horizon-resolved gradients grad_hbs
%   - gradients averaged over:
%         (i) local horizon
%         (ii) forecast replications
%     but NOT averaged over inner samples
%
% Inputs:
%   beta      : N x 1 portfolio weights
%   fc_blk    : output of msv_forecast_states_only_by_blocks
%   model     : must contain Givset
%   gamma     : risk aversion scalar
%   blockSize : passed to Sigmar_times_beta_exact
%
% Outputs:
%   grad_by_inner        : N x B_inner
%   grad_xi_by_inner     : N x B_inner
%   grad_hbs             : N x H_local x B_inner x S_fc
%   mu_by_inner          : N x B_inner
%   SigmarBeta_by_inner  : N x B_inner

    if nargin < 5 || isempty(blockSize)
        blockSize = 512;
    end

    beta = beta(:);
    N = length(beta);

    H_local = fc_blk.localHorizon;
    B_inner = fc_blk.B_inner;
    S_fc = fc_blk.nForecastPerInner;

    % Allocate outputs
    grad_hbs = zeros(N, H_local, B_inner, S_fc);

    mu_by_inner = zeros(N, B_inner);
    SigmarBeta_by_inner = zeros(N, B_inner);
    grad_by_inner = zeros(N, B_inner);
    grad_xi_by_inner = zeros(N, B_inner);

    % Loop over inner samples
    for b = 1:B_inner
        LW_b = fc_blk.static.LW(:,:,b);
        sig2_b = fc_blk.static.sigma2(b);

        mu_acc_b = zeros(N,1);
        sb_acc_b = zeros(N,1);

        for s = 1:S_fc
            mu_acc_s = zeros(N,1);
            sb_acc_s = zeros(N,1);

            for hh = 1:H_local
                omegas = fc_blk.omegas_fc(:, hh, b, s);
                lambdas = fc_blk.lambdas_fc(:, hh, b, s);

                [Sigmar_beta, mu_r] = Sigmar_times_beta_exact( ...
                    beta, LW_b, omegas, lambdas, sig2_b, model.Givset, blockSize);

                Sigmar_beta = Sigmar_beta(:);
                mu_r = mu_r(:);

                g = mu_r - gamma * Sigmar_beta;

                grad_hbs(:, hh, b, s) = g;

                mu_acc_s = mu_acc_s + mu_r;
                sb_acc_s = sb_acc_s + Sigmar_beta;
            end

            % Average over local horizon for this forecast replication
            mu_acc_b = mu_acc_b + mu_acc_s / H_local;
            sb_acc_b = sb_acc_b + sb_acc_s / H_local;
        end

        % Average over forecast replications
        mu_by_inner(:, b) = mu_acc_b / S_fc;
        SigmarBeta_by_inner(:, b) = sb_acc_b / S_fc;

        grad_by_inner(:, b) = mu_by_inner(:, b) - gamma * SigmarBeta_by_inner(:, b);

        % Transformed gradient in xi-space, one per inner sample
        gbar = grad_by_inner(:, b);
        grad_xi_by_inner(:, b) = -beta .* (beta' * gbar) + beta .* gbar;
    end
end