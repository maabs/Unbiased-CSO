function forecast_output = msv_forecast_demoToy(samples, mcmcoptions, model, staticMode)
% Forecast MSV with mu_t = 0 (no VAR).
%
% Inputs:
%   samples    : output of mcmcTrain.m (ideally already post-processed like demoToy.m)
%   mcmcoptions: used only for compatibility (optional)
%   model      : must contain N,K,T,horizon,L,Givset and model.actual (N x horizon)
%   staticMode : 'perSample' or 'staticMean'
%
% Output:
%   forecast_output.Density      : horizon x N x S simulated returns
%   forecast_output.SigmaObsMean : N x N x horizon posterior mean forecast covariance
%   forecast_output.PointF       : horizon x N point forecast (mean of Density over S)
%   forecast_output.Ferror       : horizon x N (PointF - actual')
%
if nargin < 4
    staticMode = 'perSample'; % default
end

% Number of predictive draws
if isfield(mcmcoptions,'train') && isfield(mcmcoptions.train,'T') && isfield(mcmcoptions.train,'StoreEvery')
    ForNSim = mcmcoptions.train.T / mcmcoptions.train.StoreEvery;
    ForNSim = min(ForNSim, size(samples.F,1));
else
    ForNSim = size(samples.F,1);
end

H = model.horizon;
N = model.N;
K = model.K;

% Ensure hs and deltas exist (demoToy does this post-processing)
if ~isfield(samples,'hs') || ~isfield(samples,'deltas')
    % reconstruct from samples.F
    KT = model.K*model.T;
    tildeKT = model.tildeK*model.T;
    samples.hs     = zeros(model.K, model.T, ForNSim);
    samples.deltas = zeros(model.tildeK, model.T, ForNSim);
    for s=1:ForNSim
        samples.hs(:,:,s)     = reshape(samples.F(s,1:KT), model.T, model.K)';
        samples.deltas(:,:,s) = reshape(samples.F(s,KT+1:KT+tildeKT), model.T, model.tildeK)';
    end
end

% Allocate outputs
Density      = zeros(H, N, ForNSim);
SigmaObsMean = zeros(N, N, H);
 
% ---------- Build static-mean parameters if requested ----------
if strcmpi(staticMode,'staticMean')
    phi_h_bar        = mean(samples.Phi_h(1:ForNSim,:), 1);
    phi_delta_bar    = mean(samples.Phi_delta(1:ForNSim,:), 1);
    h0_bar           = mean(samples.h_0(1:ForNSim,:), 1);
    delta0_bar       = mean(samples.delta_0(1:ForNSim,:), 1);
    sigma2_h_bar     = mean(samples.sigma2_h(1:ForNSim,:), 1);
    sigma2_delta_bar = mean(samples.sigma2_delta(1:ForNSim,:), 1);

    Weights_bar = mean(samples.Weights(:,:,1:ForNSim), 3);
    LW_bar      = model.L .* Weights_bar;
    

    if size(samples.sigma2,1) >= ForNSim
        sigma2_bar = mean(samples.sigma2(1:ForNSim,:), 1); % works for both scalar and diagonal (row vector)
    else
        sigma2_bar = mean(samples.sigma2, 2);
    end
end
LogScore = NaN(H, ForNSim);     % per-draw log N(y_actual;0,SigmaObs)
% ---------- Main predictive loop ----------
for it = 1:ForNSim

    % Choose static parameters according to the mode
    switch lower(staticMode)
        case 'persample'
            phi_h        = samples.Phi_h(it,:);
            phi_delta    = samples.Phi_delta(it,:);
            h_0          = samples.h_0(it,:);
            delta_0      = samples.delta_0(it,:);
            sigma2_h     = samples.sigma2_h(it,:);
            sigma2_delta = samples.sigma2_delta(it,:);
            LW           = model.L .* samples.Weights(:,:,it);
            ss2          = samples.sigma2(it); % scalar or 1xN

        case 'staticmean'
            phi_h        = phi_h_bar;
            phi_delta    = phi_delta_bar;
            h_0          = h0_bar;
            delta_0      = delta0_bar;
            sigma2_h     = sigma2_h_bar;
            sigma2_delta = sigma2_delta_bar;
            LW           = LW_bar;
            ss2          = sigma2_bar;

        otherwise
            error('staticMode must be ''perSample'' or ''staticMean''.');
    end

    % Endpoints from THIS draw (always sample-specific endpoints)
    hLast     = samples.hs(:, model.T, it);
    deltaLast = samples.deltas(:, model.T, it);
    
    % Forecast forward H steps
    for hh = 1:H

        % --- Propagate AR(1) for h and delta (componentwise, diagonal innovations) ---
        eps_h = randn(K,1) .* sqrt(sigma2_h(:));
        hNew  = (eye(K) - diag(phi_h)) * (h_0(:)) + diag(phi_h) * hLast + eps_h;

        eps_d = randn(model.tildeK,1) .* sqrt(sigma2_delta(:));
        dNew  = (eye(model.tildeK) - diag(phi_delta)) * (delta_0(:)) + diag(phi_delta) * deltaLast + eps_d;

        hLast     = hNew;
        deltaLast = dNew;

        % --- Map delta -> omegas, build G, lambdas ---
        omegas  = (0.5*pi) * ( (exp(dNew)-1) ./ (exp(dNew)+1) );  % same transform as code
        lambdas = exp(hNew);  % K x 1

        % Build orthogonal matrix G from Givens rotations
        G = eye(K);
        
        for k = 1:size(model.Givset,1)
            Gtmp = givensmat(omegas(k), K, model.Givset(k,1), model.Givset(k,2));
            G = Gtmp * G;
        end
        SigmaFactors = G * diag(lambdas) * G';           % K x K
        
        SigmaObs     = LW * SigmaFactors * LW';          % N x N
        
        % --- Log score on realized holdout (if available) ---
        if isfield(model,'actual') && ~isempty(model.actual)
            yact = model.actual(:,hh);      % N x 1, may contain NaN
            if isscalar(ss2)
                LogScore(hh,it) = msv_logscore_caseA(yact, LW, G, lambdas, ss2);
            else
                % diagonal V case (not your mcmcTrain.m default); handle separately if needed
                LogScore(hh,it) = msv_logscore_diagV(yact, LW, G, lambdas, ss2(:));
            end
        end
        % Add observation noise
      
        if isscalar(ss2)
            SigmaObs = SigmaObs + ss2 * eye(N);
        else
            
            SigmaObs = SigmaObs + diag(ss2(:));
        end

        % Store posterior mean covariance accumulator
        SigmaObsMean(:,:,hh) = SigmaObsMean(:,:,hh) + SigmaObs;

        % --- Sample returns y_{T+hh} ~ N(0, SigmaObs) ---
        % Use Cholesky (with jitter for safety)
        C = jitterChol(SigmaObs);          % upper or lower depending on your helper; assume lower works below
        y = C' * randn(N,1);               % if C is upper; if your jitterChol returns lower, use C*randn
        Density(hh,:,it) = y';
    end
end

% Proper predictive score: log mean exp across draws (mixture of Gaussians)
LogScoreMix = NaN(H,1);
for hh=1:H
    lls = LogScore(hh,:);
    lls = lls(~isnan(lls));
    if ~isempty(lls)
        m = max(lls);
        LogScoreMix(hh) = m + log(mean(exp(lls - m)));
    end
end

forecast_output.LogScore     = LogScore;     % H x S
forecast_output.LogScoreMix  = LogScoreMix;  % H x 1
forecast_output.LogScoreAvg  = mean(LogScoreMix(~isnan(LogScoreMix)));


SigmaObsMean = SigmaObsMean / ForNSim;

% Point forecast and error
PointF = mean(Density, 3);            % horizon x N
forecast_output.Density      = Density;
forecast_output.SigmaObsMean = SigmaObsMean;
forecast_output.PointF       = PointF;

if isfield(model,'actual') && ~isempty(model.actual)
    forecast_output.Ferror = PointF - model.actual';  % model.actual is N x horizon
    forecast_output.Ferror_rel=2*(forecast_output.Ferror)./(model.actual'+PointF);
else
    forecast_output.Ferror = [];
end

end