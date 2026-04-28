function [g_xi, g_beta, info] = msv_grad_xi_from_states(outStates, model, xi, delta, staticMode)
% Compute gradient wrt xi where beta = softmax(xi), and
% f = 1/2 * [ beta' * ( (1/(H*S)) sum SigmaY ) * beta + delta * ||beta||^2 ].
% Inputs:
%   outStates : output of msv_forecast_states_only.m
%               needs hs_fc (KxHxS) OR lambdas_fc, and omegas_fc (tildeKxHxS)
%               and static struct with LW_mean/sigma2_mean (staticMean) and/or LW(:,:,s), sigma2_samp
%   model     : needs N, K, horizon, Givset
%   xi        : N x 1 unconstrained params
%   delta     : ridge parameter (scalar)
%   staticMode: 'staticMean' or 'perSample'
%
% Outputs:
%   g_xi      : N x 1 gradient wrt xi
%   g_beta    : N x 1 gradient wrt beta (= (avg SigmaY)*beta + delta*beta)
%   info      : diagnostics (beta, betaTg, avgSigBeta)

if nargin < 5 || isempty(staticMode)
    staticMode = outStates.staticMode;
end

N = model.N;
K = model.K;
H = model.horizon;

% ----- softmax -----
xi = xi(:);
xi = xi - max(xi);                 % stability
ex = exp(xi);
beta = ex / sum(ex);               % N x 1

% ----- access forecast arrays -----
% Prefer lambdas_fc if present (cheap already)
if isfield(outStates,'lambdas_fc')
    lambdas_fc = outStates.lambdas_fc;   % K x H x S
else
    lambdas_fc = exp(outStates.hs_fc);   % K x H x S
end
omegas_fc  = outStates.omegas_fc;        % tildeK x H x S

S = size(lambdas_fc,3);

% ----- static objects (LW and noise) -----
st = outStates.static;

% Prepare accumulators
avgSigBeta = zeros(N,1);

for s = 1:S

    % choose LW and noise for this draw
    switch lower(staticMode)
        case 'staticmean'
            LW  = st.LW_mean;                % N x K
            ss2 = st.sigma2_mean;            % scalar or 1xN
        case 'persample'
            if isfield(st,'LW')
                LW = st.LW(:,:,s);           % N x K
            else
                % fallback: if only Weights were stored
                LW = model.L .* st.Weights_mean; % not ideal, but prevents crash
            end
            if isfield(st,'sigma2_samp')
                ss2 = st.sigma2_samp(s,:);   % scalar or 1xN
            else
                ss2 = st.sigma2_mean;
            end
        otherwise
            error('staticMode must be staticMean or perSample');
    end

    % precompute LW' * beta once per draw if LW fixed across horizon
    % (it is fixed in your setup)
    w = LW' * beta;                          % K x 1, cost O(NK)

    for hh = 1:H
        lam = lambdas_fc(:,hh,s);            % K x 1
        omg = omegas_fc(:,hh,s);             % tildeK x 1 (= number of rotations)

        % Apply G' to w via Givens (no explicit G)
        % Apply G' to w via Givens (no explicit G)
        u = apply_givens_vec(w, omg, model.Givset, 'GT');   % K x 1
        
        % Multiply by diag(lam)
        u = lam(:) .* u;
        
        % Apply G to u
        u = apply_givens_vec(u, omg, model.Givset, 'G');    % K x 1          % K x 1

        % Map back to N-dim: LW * u
        v = LW * u;                                             % N x 1, cost O(NK)

        % Add observation noise D*beta
        if isscalar(ss2)
            v = v + ss2 * beta;
        else
            v = v + ss2(:) .* beta;                             % diag(ss2)*beta
        end

        avgSigBeta = avgSigBeta + v;
    end
end

avgSigBeta = avgSigBeta / (S*H);           % = (avg SigmaY)*beta

% Gradient wrt beta for f = 1/2[ beta' avgSigma beta + delta ||beta||^2 ]
% is: g_beta = avgSigma*beta + delta*beta
g_beta = avgSigBeta + delta * beta;

% Chain rule through softmax: g_xi = (diag(beta) - beta*beta') * g_beta
betaTg = beta' * g_beta;
g_xi = beta .* (g_beta - betaTg);

% diagnostics
info.beta = beta;
info.betaTg = betaTg;
info.avgSigBeta = avgSigBeta;
end

