function out = msv_forecast_states_only_by_blocks(samples_blk, model, outer_prev, localHorizon, nForecastPerInner)
% Forecast latent MSV states from the output of mcmcTrain_blocks.
%
% This version is designed for the blocked sampler:
%   - inner samples come from samples_blk.inner
%   - the forecasting hyperparameters come from the PREVIOUS/FROZEN outer sample
%
% Inputs:
%   samples_blk         : output of mcmcTrain_blocks
%   model               : model structure (used for dimensions and L)
%   outer_prev          : struct containing the previous/frozen outer sample:
%                         .h_0
%                         .delta_0
%                         .sigma2_h
%                         .sigma2_delta
%                         .phi_h
%                         .phi_delta
%   localHorizon        : local forecast horizon H_local
%   nForecastPerInner   : number of forecast paths per stored inner sample
%
% Output:
%   out.hs_fc           : [K x H_local x B_inner x S_fc]
%   out.deltas_fc       : [tildeK x H_local x B_inner x S_fc]
%   out.lambdas_fc      : [K x H_local x B_inner x S_fc]
%   out.omegas_fc       : [tildeK x H_local x B_inner x S_fc]
%   out.static          : struct with static quantities aligned to inner samples
%   out.localHorizon    : local forecast horizon used
%   out.B_inner         : number of stored inner samples
%   out.nForecastPerInner : number of forecast paths per inner sample

    if nargin < 5 || isempty(nForecastPerInner)
        nForecastPerInner = 1;
    end

    if nargin < 4 || isempty(localHorizon)
        error('localHorizon must be provided.');
    end

    % ------------------------------------------------------------
    % Dimensions
    % ------------------------------------------------------------
    K = model.K;
    tildeK = model.tildeK;
    T = model.T;

    B_inner = size(samples_blk.inner.F, 1);

    % ------------------------------------------------------------
    % Reconstruct hs and deltas from stored inner F
    % ------------------------------------------------------------
    KT = K * T;
    tildeKT = tildeK * T;

    hs_last = zeros(K, B_inner);
    deltas_last = zeros(tildeK, B_inner);

    for b = 1:B_inner
        Fb = samples_blk.inner.F(b, :);

        hs_b = reshape(Fb(1:KT), T, K)';
        deltas_b = reshape(Fb(KT+1:KT+tildeKT), T, tildeK)';

        hs_last(:, b) = hs_b(:, T);
        deltas_last(:, b) = deltas_b(:, T);
    end

    % ------------------------------------------------------------
    % Allocate forecast outputs
    % ------------------------------------------------------------
    hs_fc      = zeros(K, localHorizon, B_inner, nForecastPerInner);
    deltas_fc  = zeros(tildeK, localHorizon, B_inner, nForecastPerInner);
    lambdas_fc = zeros(K, localHorizon, B_inner, nForecastPerInner);
    omegas_fc  = zeros(tildeK, localHorizon, B_inner, nForecastPerInner);

    % ------------------------------------------------------------
    % Static quantities to carry forward for later gradient use
    % ------------------------------------------------------------
    static = struct();

    % Inner-sample-specific static quantities
    static.Weights = samples_blk.inner.Weights;          % [N x K x B_inner]
    static.sigma2  = samples_blk.inner.sigma2(:);       % [B_inner x 1]

    if isfield(model, 'L')
        static.LW = zeros(size(model.L,1), size(model.L,2), B_inner);
        for b = 1:B_inner
            static.LW(:,:,b) = model.L .* samples_blk.inner.Weights(:,:,b);
        end
    end

    % Previous/frozen outer hyperparameters
    static.outer_prev = outer_prev;

    % ------------------------------------------------------------
    % Forecast loop
    % ------------------------------------------------------------
    phi_h        = outer_prev.phi_h(:);
    phi_delta    = outer_prev.phi_delta(:);
    h_0          = outer_prev.h_0(:);
    delta_0      = outer_prev.delta_0(:);
    sigma2_h     = outer_prev.sigma2_h(:);
    sigma2_delta = outer_prev.sigma2_delta(:);

    for b = 1:B_inner
        for s = 1:nForecastPerInner

            hLast = hs_last(:, b);
            dLast = deltas_last(:, b);

            for hh = 1:localHorizon
                hNew = (1 - phi_h) .* h_0 + phi_h .* hLast ...
                     + randn(K,1) .* sqrt(sigma2_h);

                dNew = (1 - phi_delta) .* delta_0 + phi_delta .* dLast ...
                     + randn(tildeK,1) .* sqrt(sigma2_delta);

                hs_fc(:, hh, b, s) = hNew;
                deltas_fc(:, hh, b, s) = dNew;

                lambdas_fc(:, hh, b, s) = exp(hNew);
                omegas_fc(:, hh, b, s) = (0.5*pi) * ((exp(dNew) - 1) ./ (exp(dNew) + 1));

                hLast = hNew;
                dLast = dNew;
            end
        end
    end

    % ------------------------------------------------------------
    % Pack output
    % ------------------------------------------------------------
    out = struct();
    out.hs_fc = hs_fc;
    out.deltas_fc = deltas_fc;
    out.lambdas_fc = lambdas_fc;
    out.omegas_fc = omegas_fc;

    out.static = static;

    out.localHorizon = localHorizon;
    out.B_inner = B_inner;
    out.nForecastPerInner = nForecastPerInner;
end