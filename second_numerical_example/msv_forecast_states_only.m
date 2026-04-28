function out = msv_forecast_states_only(samples, model, staticMode, ForNSim)
% Forecast only (hs, deltas) forward; return static parameters used
% plus LW and sigma2 means.
%
% Output:
%   out.hs_fc, out.deltas_fc, out.lambdas_fc, out.omegas_fc
%   out.static: struct with:
%       - (phi_h, phi_delta, h_0, delta_0, sigma2_h, sigma2_delta)
%       - LW_mean, sigma2_mean   (always present if fields exist)
%       - if perSample: LW(:,:,s) and sigma2_samp(s,:) (optional)

if nargin < 3 || isempty(staticMode)
    staticMode = 'perSample';
end

H = model.horizon;
K = model.K;
tildeK = model.tildeK;

if nargin < 4 || isempty(ForNSim)
    ForNSim = size(samples.F,1);
else
    ForNSim = min(ForNSim, size(samples.F,1));
end

% Ensure hs/deltas exist
if ~isfield(samples,'hs') || ~isfield(samples,'deltas')
    KT = model.K * model.T;
    tildeKT = model.tildeK * model.T;
    samples.hs     = zeros(model.K, model.T, ForNSim);
    samples.deltas = zeros(model.tildeK, model.T, ForNSim);
    for s = 1:ForNSim
        samples.hs(:,:,s)     = reshape(samples.F(s,1:KT), model.T, model.K)';
        samples.deltas(:,:,s) = reshape(samples.F(s,KT+1:KT+tildeKT), model.T, model.tildeK)';
    end
end

% Allocate outputs
hs_fc      = zeros(K, H, ForNSim);
deltas_fc  = zeros(tildeK, H, ForNSim);
lambdas_fc = zeros(K, H, ForNSim);
omegas_fc  = zeros(tildeK, H, ForNSim);

static = struct();

% ---------- Always compute and store means for LW and sigma2 (if available) ----------
% LW_mean = L .* mean(Weights, 3)
if isfield(samples,'Weights') && isfield(model,'L')
    static.Weights_mean = mean(samples.Weights(:,:,1:ForNSim), 3);     % N x K
    static.LW_mean      = model.L .* static.Weights_mean;              % N x K
end

% sigma2_mean: handle scalar noise or diagonal noise
if isfield(samples,'sigma2')
    s2 = samples.sigma2;

    % common cases:
    % 1) scalar per draw: size = [S,1] or [1,S]
    % 2) diagonal per draw: size = [S,N]
    if isvector(s2)
        static.sigma2_mean = mean(s2(1:ForNSim));                      % scalar
    else
        % assume [S, N] or [N, S]; pick orientation by matching model.N if possible
        if isfield(model,'N') && size(s2,2) == model.N
            static.sigma2_mean = mean(s2(1:ForNSim,:), 1);             % 1 x N
        elseif isfield(model,'N') && size(s2,1) == model.N
            static.sigma2_mean = mean(s2(:,1:ForNSim), 2)';            % 1 x N
        else
            % fallback: mean over first dim
            static.sigma2_mean = mean(s2(1:ForNSim,:), 1);
        end
    end
end

% ---------- Static parameter means (only if requested) ----------
if strcmpi(staticMode,'staticMean')
    static.phi_h        = mean(samples.Phi_h(1:ForNSim,:), 1);
    static.phi_delta    = mean(samples.Phi_delta(1:ForNSim,:), 1);
    static.h_0          = mean(samples.h_0(1:ForNSim,:), 1);
    static.delta_0      = mean(samples.delta_0(1:ForNSim,:), 1);
    static.sigma2_h     = mean(samples.sigma2_h(1:ForNSim,:), 1);
    static.sigma2_delta = mean(samples.sigma2_delta(1:ForNSim,:), 1);
end

% ---------- If perSample, store per-draw static params (and optionally LW, sigma2) ----------
if strcmpi(staticMode,'perSample')
    static.phi_h        = zeros(ForNSim, K);
    static.phi_delta    = zeros(ForNSim, tildeK);
    static.h_0          = zeros(ForNSim, K);
    static.delta_0      = zeros(ForNSim, tildeK);
    static.sigma2_h     = zeros(ForNSim, K);
    static.sigma2_delta = zeros(ForNSim, tildeK);

    % optional: store per-draw LW and sigma2 used
    if isfield(samples,'Weights') && isfield(model,'L')
        static.LW = zeros(size(model.L,1), size(model.L,2), ForNSim);   % N x K x S
    end
    if isfield(samples,'sigma2')
        % store sigma2 draws in consistent shape [S, ?]
        if isvector(samples.sigma2)
            static.sigma2_samp = zeros(ForNSim, 1);
        else
            % try [S,N]
            if isfield(model,'N') && size(samples.sigma2,2) == model.N
                static.sigma2_samp = zeros(ForNSim, model.N);
            else
                static.sigma2_samp = zeros(ForNSim, size(samples.sigma2,2));
            end
        end
    end
end

% ---------- Main loop ----------
for s = 1:ForNSim

    if strcmpi(staticMode,'perSample')
        phi_h        = samples.Phi_h(s,:);
        phi_delta    = samples.Phi_delta(s,:);
        h_0          = samples.h_0(s,:);
        delta_0      = samples.delta_0(s,:);
        sigma2_h     = samples.sigma2_h(s,:);
        sigma2_delta = samples.sigma2_delta(s,:);

        static.phi_h(s,:)        = phi_h;
        static.phi_delta(s,:)    = phi_delta;
        static.h_0(s,:)          = h_0;
        static.delta_0(s,:)      = delta_0;
        static.sigma2_h(s,:)     = sigma2_h;
        static.sigma2_delta(s,:) = sigma2_delta;

        if isfield(samples,'Weights') && isfield(model,'L')
            static.LW(:,:,s) = model.L .* samples.Weights(:,:,s);
        end
        if isfield(samples,'sigma2')
            if isvector(samples.sigma2)
                static.sigma2_samp(s,1) = samples.sigma2(s);
            else
                if isfield(model,'N') && size(samples.sigma2,2) == model.N
                    static.sigma2_samp(s,:) = samples.sigma2(s,:);
                elseif isfield(model,'N') && size(samples.sigma2,1) == model.N
                    static.sigma2_samp(s,:) = samples.sigma2(:,s)';
                else
                    static.sigma2_samp(s,:) = samples.sigma2(s,:);
                end
            end
        end

    else
        phi_h        = static.phi_h;
        phi_delta    = static.phi_delta;
        h_0          = static.h_0;
        delta_0      = static.delta_0;
        sigma2_h     = static.sigma2_h;
        sigma2_delta = static.sigma2_delta;
    end

    % endpoints
    hLast     = samples.hs(:, model.T, s);
    deltaLast = samples.deltas(:, model.T, s);

    for hh = 1:H
        hNew = (1 - phi_h(:)) .* h_0(:) + phi_h(:) .* hLast ...
             + randn(K,1) .* sqrt(sigma2_h(:));

        dNew = (1 - phi_delta(:)) .* delta_0(:) + phi_delta(:) .* deltaLast ...
             + randn(tildeK,1) .* sqrt(sigma2_delta(:));

        hs_fc(:,hh,s)     = hNew;
        deltas_fc(:,hh,s) = dNew;

        lambdas_fc(:,hh,s) = exp(hNew);
        omegas_fc(:,hh,s)  = (0.5*pi) * ((exp(dNew)-1)./(exp(dNew)+1));

        hLast     = hNew;
        deltaLast = dNew;
    end
end

out.hs_fc      = hs_fc;
out.deltas_fc  = deltas_fc;
out.lambdas_fc = lambdas_fc;
out.omegas_fc  = omegas_fc;
out.static     = static;
out.staticMode = staticMode;
out.ForNSim    = ForNSim;

end