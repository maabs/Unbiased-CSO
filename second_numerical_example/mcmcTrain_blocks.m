function [model, samples, accRates] = mcmcTrain_blocks(model, PropDist, trainOps, Langevin, B_inner)
% Inputs:
%         -- model: structure containing likelihood, latent-state, and prior parameters
%         -- PropDist: structure defining proposal distribution parameters
%         -- trainOps: options structure; here we use trainOps.Burnin for the inner burn-in
%         -- Langevin: 1 if Langevin/MALA-type proposals are used for the latent block, 0 otherwise
%         -- B_inner: number of post-burn-in inner iterations to store
%
% Outputs:
%         -- model: updated model after the full blocked transition
%         -- samples.inner: stores only inner-block variables over the last B_inner inner iterations
%         -- samples.outer: stores only outer-block variables after the single outer update
%         -- accRates: acceptance rates for inner blocks over all inner iterations including burn-in

BurnInIters = trainOps.Burnin;
num_inner_total = BurnInIters + B_inner;

% Total dimension of the stacked latent block F = [vec(hs'); vec(deltas')]
n = model.K * model.T + model.tildeK * model.T;

% ---------------------------------
% STORAGE: INNER BLOCK
% ---------------------------------
samples.inner = struct();

samples.inner.F       = zeros(B_inner, n);
samples.inner.Ft      = zeros(model.K, model.T, B_inner);
samples.inner.sigma2  = zeros(1, B_inner);
samples.inner.Weights = zeros(model.N, model.K, B_inner);
samples.inner.LogL    = zeros(1, B_inner);

% Optional: if later you decide you want these explicitly stored instead of
% reconstructing them from F, uncomment the following:
%
% samples.inner.hs     = zeros(model.K, model.T, B_inner);
% samples.inner.deltas = zeros(model.tildeK, model.T, B_inner);
% samples.inner.omegas = zeros(model.tildeK, model.T, B_inner);

% ---------------------------------
% STORAGE: OUTER BLOCK
% ---------------------------------
samples.outer = struct();

samples.outer.h_0            = zeros(1, model.K);
samples.outer.delta_0        = zeros(1, model.tildeK);
samples.outer.sigma2_h       = zeros(1, model.K);
samples.outer.sigma2_delta   = zeros(1, model.tildeK);
samples.outer.tildephi_h     = zeros(1, model.K);
samples.outer.tildephi_delta = zeros(1, model.tildeK);
samples.outer.phi_h          = zeros(1, model.K);
samples.outer.phi_delta      = zeros(1, model.tildeK);

% Optional debugging / diagnostics fields:
%
% samples.outer.oldLogLphi   = [];
% samples.outer.oldPriorphi  = [];

% ---------------------------------
% COUNTERS / ACCEPTANCE ACCUMULATORS
% ---------------------------------
cnt_inner = 0;
acceptF = 0;
acceptFt = zeros(model.T, 1);

% accRates.Phi is intentionally omitted in this blocked version
accRates = struct();


% ---------------------------------
% BUILD SPARSITY STRUCTURE FOR THE AR PRECISION MATRICES
% ---------------------------------
iAll = [];
jAll = [];
en = 0;

for k = 1:model.K
    iAll = [iAll, (1+en):(model.T+en)];
    jAll = [jAll, (1+en):(model.T+en)];

    iAll = [iAll, (1+en):(model.T-1+en)];
    jAll = [jAll, (2+en):(model.T+en)];

    iAll = [iAll, (2+en):(model.T+en)];
    jAll = [jAll, (1+en):(model.T-1+en)];

    en = en + model.T;
end

for k = 1:model.tildeK
    iAll = [iAll, (1+en):(model.T+en)];
    jAll = [jAll, (1+en):(model.T+en)];

    iAll = [iAll, (1+en):(model.T-1+en)];
    jAll = [jAll, (2+en):(model.T+en)];

    iAll = [iAll, (2+en):(model.T+en)];
    jAll = [jAll, (1+en):(model.T-1+en)];

    en = en + model.T;
end

% ---------------------------------
% TRANSFORM TILDE-PHI TO PHI
% ---------------------------------
model.phi_h = (exp(model.tildephi_h) - 1) ./ (exp(model.tildephi_h) + 1);
model.phi_delta = (exp(model.tildephi_delta) - 1) ./ (exp(model.tildephi_delta) + 1);

% ---------------------------------
% BUILD INITIAL PRIOR PRECISION MATRIX W
% ---------------------------------
numBlocks = model.K + model.tildeK;
nnz_total = numBlocks * (3*model.T - 2);

nonZeros = zeros(1, nnz_total);
idx = 1;

for i = 1:model.K
    d = ((1 + model.phi_h(i)^2) / model.sigma2_h(i)) * ones(1, model.T);
    d(1) = 1 / model.sigma2_h(i);
    d(end) = 1 / model.sigma2_h(i);

    nonZeros(idx:idx+model.T-1) = d;
    idx = idx + model.T;

    offd = -model.phi_h(i) / model.sigma2_h(i);
    len = 2*model.T - 2;
    nonZeros(idx:idx+len-1) = offd;
    idx = idx + len;
end

for j = 1:model.tildeK
    d = ((1 + model.phi_delta(j)^2) / model.sigma2_delta(j)) * ones(1, model.T);
    d(1) = 1 / model.sigma2_delta(j);
    d(end) = 1 / model.sigma2_delta(j);

    nonZeros(idx:idx+model.T-1) = d;
    idx = idx + model.T;

    offd = -model.phi_delta(j) / model.sigma2_delta(j);
    len = 2*model.T - 2;
    nonZeros(idx:idx+len-1) = offd;
    idx = idx + len;
end

W = sparse(iAll, jAll, nonZeros);

Lfree = jitterChol(W);
Wonly = W;
W = W + diag(sparse((1./model.auxLikVar) * ones(1, n)));
L = jitterChol(W)';

% ---------------------------------
% BUILD INITIAL STACKED LATENT VECTOR F
% ---------------------------------
tmp = model.hs';
F = tmp(:);

tmp = model.deltas';
F = [F; tmp(:)];

% ---------------------------------
% BUILD INITIAL MEAN VECTOR mu
% ---------------------------------
tmp = repmat(model.h_0, model.T, 1);
mu = tmp(:);

tmp = repmat(model.delta_0, model.T, 1);
mu = [mu; tmp(:)];

% ---------------------------------
% INITIAL LOG-DENSITY OF LATENT BLOCK UNDER CURRENT PHI
% ---------------------------------
ok = Lfree * (F - mu);
oldLogLphi = -0.5 * (model.T * (model.K + model.tildeK)) * log(2*pi) ...
             + sum(log(diag(Lfree))) ...
             - 0.5 * (ok' * ok);

% ---------------------------------
% INITIAL PRIOR ON TILDE-PHI
% ---------------------------------
if strcmp(model.exchangeablePriorphi, 'yes')
    oldPriorphi_h = logmarginalizedNormalGam(model.tildephi_h, ...
        model.priorPhi_h.mu0, model.priorPhi_h.k0, ...
        model.priorPhi_h.alpha0, model.priorPhi_h.beta0);

    oldPriorphi_delta = logmarginalizedNormalGam(model.tildephi_delta, ...
        model.priorPhi_delta.mu0, model.priorPhi_delta.k0, ...
        model.priorPhi_delta.alpha0, model.priorPhi_delta.beta0);
else
    oldPriorphi_h = logNormal(model.tildephi_h, ...
        model.priorPhi_h.mu0, model.priorPhi_h.s2);

    oldPriorphi_delta = logNormal(model.tildephi_delta, ...
        model.priorPhi_delta.mu0, model.priorPhi_delta.s2);
end

oldPriorphi = oldPriorphi_h + oldPriorphi_delta;

% ---------------------------------
% NUMBER OF OBSERVED ENTRIES
% ---------------------------------
num_OfNonNans = sum(~isnan(model.Y(:)));
model.num_OfNonNans = num_OfNonNans;

% ---------------------------------
% GRADIENT STORAGE FOR LANGEVIN BLOCK
% ---------------------------------
if Langevin == 1
    gradDeltas = zeros(size(model.deltas));
    gradHs = zeros(size(model.hs));

    derF = zeros(length(F), 1);
    derFnew = zeros(length(F), 1);
end

% ---------------------------------
% INNER LOOP: UPDATE Ft, Weights, sigma2, and F=(hs,deltas)
% ---------------------------------
for it = 1:num_inner_total

        % =========================================================
    % STEP 1: UPDATE Ft
    % Setting assumed here:
    %   model.useFactorModel       = 'yes'
    %   model.sampleFactorsByGibbs = 'no'
    %   model.diagonalSigmat       = 'no'
    % =========================================================
    for t = 1:model.T
        ind = find(~isnan(model.Y(:,t)));

        if length(ind) > 0
            lognewlambdas = -model.hs(:,t) - log(model.deltaFactors(t)) ...
                            + log(2*exp(model.hs(:,t)) + model.deltaFactors(t));

            Lobs = model.L(ind,:) .* model.Weights(ind,:);
            rt = model.Y(ind,t) - Lobs * model.Ft(:,t);
            gradFt = (1/model.sigma2) * (Lobs' * rt);

            Zt = model.Ft(:,t) ...
                 + (model.deltaFactors(t)/2) * gradFt ...
                 + sqrt(model.deltaFactors(t)/2) * randn(model.K,1);

            % Precompute trigonometric values for this time t
            c_all = cos(model.omegas(:,t));
            s_all = sin(model.omegas(:,t));

            % Forward Givens sweep
            v = Zt;
            for k = 1:size(model.Givset,1)
                i = model.Givset(k,1);
                j = model.Givset(k,2);

                c = c_all(k);
                s = s_all(k);

                vi = v(i);
                vj = v(j);

                v(i) =  c*vi - s*vj;
                v(j) =  s*vi + c*vj;
            end

            v = v .* exp(-0.5 * lognewlambdas);

            keep = randn(model.K,1);
            v = ((2/model.deltaFactors(t)) * v + keep) .* exp(-0.5 * lognewlambdas);

            % Backward Givens sweep
            for k = size(model.Givset,1):-1:1
                i = model.Givset(k,1);
                j = model.Givset(k,2);

                c = c_all(k);
                s = s_all(k);

                vi = v(i);
                vj = v(j);

                v(i) =  c*vi + s*vj;
                v(j) = -s*vi + c*vj;
            end

            Ftnew = v;

            rtnew = model.Y(ind,t) - Lobs * Ftnew;
            gradFtnew = (1/model.sigma2) * (Lobs' * rtnew);

            oldlikft = -(0.5/model.sigma2) * (rt' * rt);
            newlikft = -(0.5/model.sigma2) * (rtnew' * rtnew);

            corrFactor = -(Zt' - model.Ft(:,t)') * gradFt ...
                         + (Zt' - Ftnew') * gradFtnew;
            corrFactor = corrFactor ...
                         - (model.deltaFactors(t)/4) * (gradFtnew' * gradFtnew - gradFt' * gradFt);

            [accept, uprob] = metropolisHastings(newlikft + corrFactor, oldlikft, 0, 0);

            % In the blocked version, report inner acceptance over ALL inner iterations
            acceptFt(t) = acceptFt(t) + accept;

            if accept == 1
                model.Ft(:,t) = Ftnew;
            end

        else
            % If no observations at time t, sample Ft(:,t) from its prior
            v = randn(model.K,1) .* exp(0.5 * model.hs(:,t));

            c_all = cos(model.omegas(:,t));
            s_all = sin(model.omegas(:,t));

            for k = size(model.Givset,1):-1:1
                i = model.Givset(k,1);
                j = model.Givset(k,2);

                c = c_all(k);
                s = s_all(k);

                vi = v(i);
                vj = v(j);

                v(i) =  c*vi + s*vj;
                v(j) = -s*vi + c*vj;
            end

            model.Ft(:,t) = v;
        end
    end


        % =========================================================
    % STEP 2: UPDATE Weights
    % Gaussian Gibbs update, row by row
    % =========================================================
    for nnn = 1:model.N
        ind = find(~isnan(model.Y(nnn,:)));
        Yn = model.Y(nnn, ind);

        Phi = model.L(nnn,:)' .* model.Ft(:,ind);
        PPhi = Phi * Phi';
        Smatrix = PPhi + (model.sigma2 / model.sigma2weights) * eye(model.K);

        Lmatrix = jitterChol(Smatrix);

        tmp = sqrt(model.sigma2) * (Lmatrix \ randn(model.K,1)) ...
              + (Lmatrix \ (Lmatrix' \ (Phi * Yn(:))));

        model.Weights(nnn,:) = tmp';
    end

        % =========================================================
    % STEP 3: UPDATE sigma2
    % Inverse-Gamma Gibbs update for observation noise variance
    % =========================================================
    alpha_post = model.priorSigma2.alpha0 + num_OfNonNans / 2;
    beta_post = model.priorSigma2.beta0 ...
                + 0.5 * nansum(nansum((model.Y - (model.L .* model.Weights) * model.Ft).^2));

    model.sigma2 = 1 / gamrnd(alpha_post, 1 / beta_post);

    % Optional numerical safeguard:
    % model.sigma2(model.sigma2 < 1e-10) = 1e-10;

        % =========================================================
    % STEP 4: COMPUTE OLD LOGLIK / GRADIENTS FOR F=(hs,deltas)
    % =========================================================
    oldloglik = 0;

    if Langevin == 1
        for t = 1:model.T
            [tmp1, tmp2, tmpgradOmega, tmpgradHs] = msv_loglik( ...
                model.Ft(:,t), model.omegas(:,t), model.hs(:,t), model.Givset);

            gradDeltas(:,t) = tmpgradOmega';
            gradDeltas(:,t) = gradDeltas(:,t) .* ...
                (pi * (exp(model.deltas(:,t)) ./ ((1 + exp(model.deltas(:,t))).^2)));

            gradHs(:,t) = tmpgradHs';
            oldloglik = oldloglik + tmp1;
        end
    else
        for t = 1:model.T
            oldloglik = oldloglik + msv_loglik( ...
                model.Ft(:,t), model.omegas(:,t), model.hs(:,t), model.Givset);
        end
    end

    storeloglik = oldloglik;

        % =========================================================
    % STEP 5: UPDATE F = (hs,deltas)
    % Joint Langevin / MH update of the latent block
    % =========================================================

    if Langevin == 1
        % old gradient
        tmp = gradHs';
        derF(1:model.K*model.T) = tmp(:);

        tmp = gradDeltas';
        derF((model.K*model.T+1):end) = tmp(:);

        Z = F + (model.auxLikVar .* derF) + sqrt(model.auxLikVar) .* randn(n,1);
    else
        Z = F + sqrt(model.auxLikVar) * randn(n,1);
    end

    % Propose new values for hs and deltas
    Fnew = L' \ (L \ ((1./model.auxLikVar) * Z + Wonly * mu) + randn(n,1));

    Fnew(Fnew < -100) = -100;
    Fnew(Fnew > 100) = 100;

    if strcmp(model.diagonalSigmat, 'yes') == 1
        Fnew((model.K*model.T+1):end) = 0;
    end

    hs = reshape(Fnew(1:model.K*model.T), model.T, model.K)';
    deltas = reshape(Fnew((model.K*model.T+1):end), model.T, model.tildeK)';
    omegas = (0.5*pi) * ((exp(deltas)-1) ./ (exp(deltas) + 1));

    % Evaluate proposed log-likelihood and gradients
    newloglik = 0;

    if Langevin == 1
        for t = 1:model.T
            [tmp1, tmp2, tmpgradOmega, tmpgradHs] = msv_loglik( ...
                model.Ft(:,t), omegas(:,t), hs(:,t), model.Givset);

            gradDeltas(:,t) = tmpgradOmega';
            gradDeltas(:,t) = gradDeltas(:,t) .* ...
                (pi * (exp(deltas(:,t)) ./ ((1 + exp(deltas(:,t))).^2)));

            gradHs(:,t) = tmpgradHs';
            newloglik = newloglik + tmp1;
        end
    else
        for t = 1:model.T
            newloglik = newloglik + msv_loglik( ...
                model.Ft(:,t), omegas(:,t), hs(:,t), model.Givset);
        end
    end

    % Metropolis-Hastings correction
    corrFactor = 0;

    if Langevin == 1
        % new gradient
        tmp = gradHs';
        derFnew(1:model.K*model.T) = tmp(:);

        tmp = gradDeltas';
        derFnew((model.K*model.T+1):end) = tmp(:);

        corrFactor = - (Z - F)' * derF + (Z - Fnew)' * derFnew;
        corrFactor = corrFactor ...
            - (model.auxLikVar(1)/2) * (derFnew' * derFnew - derF' * derF);
    end

    [accept, uprob] = metropolisHastings(newloglik + corrFactor, oldloglik, 0, 0);

    % In the blocked version, report inner acceptance over ALL inner iterations
    acceptF = acceptF + accept;

    if accept == 1
        F = Fnew;
        model.hs = hs;
        model.deltas = deltas;
        model.omegas = omegas;

        storeloglik = newloglik;

        ok = Lfree * (F - mu);
        oldLogLphi = -0.5 * (model.T * (model.K + model.tildeK)) * log(2*pi) ...
                     + sum(log(diag(Lfree))) ...
                     - 0.5 * (ok' * ok);
    end

    % =========================================================
    % STORE INNER VARIABLES (ONLY AFTER INNER BURN-IN)
    % =========================================================
    if it > BurnInIters
        cnt_inner = cnt_inner + 1;

        samples.inner.F(cnt_inner,:) = F;
        samples.inner.Ft(:,:,cnt_inner) = model.Ft;
        samples.inner.sigma2(cnt_inner) = model.sigma2;
        samples.inner.Weights(:,:,cnt_inner) = model.Weights;

        if strcmp(model.useFactorModel, 'yes')
            samples.inner.LogL(cnt_inner) = ...
                - (0.5 * num_OfNonNans) * log(2*pi*model.sigma2) ...
                - nansum(nansum((model.Y - (model.L .* model.Weights) * model.Ft).^2)) ...
                  / (2 * model.sigma2);
        else
            samples.inner.LogL(cnt_inner) = storeloglik;
        end

        % Optional explicit storage of hs / deltas / omegas if you later want it:
        %
        % samples.inner.hs(:,:,cnt_inner)     = model.hs;
        % samples.inner.deltas(:,:,cnt_inner) = model.deltas;
        % samples.inner.omegas(:,:,cnt_inner) = model.omegas;
    end

end

% ---------------------------------
% RECONSTRUCT FROM FINAL INNER STATE
% ---------------------------------
model.hs = reshape(F(1:model.K * model.T), model.T, model.K)';
model.lambdas = exp(model.hs);

model.deltas = reshape(F((model.K * model.T + 1):end), model.T, model.tildeK)';
model.omegas = (0.5 * pi) * ((exp(model.deltas) - 1) ./ (exp(model.deltas) + 1));

% ---------------------------------
% OUTER UPDATE: h_0, delta_0
% ---------------------------------
% Current version follows the deterministic plug-in updates from mcmcTrain.m

t1 = 1 - model.phi_h;
t2 = 1 - model.phi_h.^2;

model.h_0 = t2 .* (model.hs(:,1)') ...
    + t1 .* (sum(model.hs(:,2:end) ...
    - repmat(model.phi_h', 1, model.T-1) .* model.hs(:,1:end-1), 2)');
model.h_0 = model.h_0 ./ (t2 + (model.T-1) * (t1.^2));

t1 = 1 - model.phi_delta;
t2 = 1 - model.phi_delta.^2;

model.delta_0 = t2 .* (model.deltas(:,1)') ...
    + t1 .* (sum(model.deltas(:,2:end) ...
    - repmat(model.phi_delta', 1, model.T-1) .* model.deltas(:,1:end-1), 2)');
model.delta_0 = model.delta_0 ./ (t2 + (model.T-1) * (t1.^2));

% ---------------------------------
% REBUILD mu
% ---------------------------------
tmp = repmat(model.h_0, model.T, 1);
mu = tmp(:);

tmp = repmat(model.delta_0, model.T, 1);
mu = [mu; tmp(:)];

% ---------------------------------
% OUTER UPDATE: sigma2_h, sigma2_delta
% ---------------------------------
newSigmar = 0.5 * (model.T + model.priorSigma2_h.sigmar);
newSsigma = model.priorSigma2_h.Ssigma ...
            + (1 - model.phi_h.^2) .* ((model.hs(:,1)' - model.h_0).^2);

hhs = model.hs - repmat(model.h_0', 1, model.T);
newSsigma = newSsigma ...
            + sum((hhs(:,2:end) - repmat(model.phi_h', 1, model.T-1) .* hhs(:,1:end-1)).^2, 2)';
newSsigma = 0.5 * newSsigma;

model.sigma2_h = 1 ./ gamrnd(newSigmar, 1 ./ newSsigma);

newSigmar = 0.5 * (model.T + model.priorSigma2_delta.sigmar);
newSsigma = model.priorSigma2_delta.Ssigma ...
            + (1 - model.phi_delta.^2) .* ((model.deltas(:,1)' - model.delta_0).^2);

ddeltas = model.deltas - repmat(model.delta_0', 1, model.T);
newSsigma = newSsigma ...
            + sum((ddeltas(:,2:end) - repmat(model.phi_delta', 1, model.T-1) .* ddeltas(:,1:end-1)).^2, 2)';
newSsigma = 0.5 * newSsigma;

model.sigma2_delta = 1 ./ gamrnd(newSigmar, 1 ./ newSsigma);

% ---------------------------------
% REBUILD W, Wonly, Lfree, L
% ---------------------------------
numBlocks = model.K + model.tildeK;
nnz_total = numBlocks * (3 * model.T - 2);

nonZeros = zeros(1, nnz_total);
idx = 1;

for i = 1:model.K
    d = ((1 + model.phi_h(i)^2) / model.sigma2_h(i)) * ones(1, model.T);
    d(1) = 1 / model.sigma2_h(i);
    d(end) = 1 / model.sigma2_h(i);

    nonZeros(idx:idx+model.T-1) = d;
    idx = idx + model.T;

    offd = -model.phi_h(i) / model.sigma2_h(i);
    len = 2 * model.T - 2;
    nonZeros(idx:idx+len-1) = offd;
    idx = idx + len;
end

for j = 1:model.tildeK
    d = ((1 + model.phi_delta(j)^2) / model.sigma2_delta(j)) * ones(1, model.T);
    d(1) = 1 / model.sigma2_delta(j);
    d(end) = 1 / model.sigma2_delta(j);

    nonZeros(idx:idx+model.T-1) = d;
    idx = idx + model.T;

    offd = -model.phi_delta(j) / model.sigma2_delta(j);
    len = 2 * model.T - 2;
    nonZeros(idx:idx+len-1) = offd;
    idx = idx + len;
end

W = sparse(iAll, jAll, nonZeros);

Lfree = jitterChol(W);
Wonly = W;
W = W + diag(sparse((1 ./ model.auxLikVar) * ones(1, n)));
L = jitterChol(W)';

% Update oldLogLphi since mu, sigma2_h, sigma2_delta changed
    ok = Lfree * (F - mu);
oldLogLphi = -0.5 * (model.T * (model.K + model.tildeK)) * log(2*pi) ...
             + sum(log(diag(Lfree))) ...
             - 0.5 * (ok' * ok);

% ---------------------------------
% OUTER UPDATE: phi
% ---------------------------------
newtildePhi = randn(1, model.K + model.tildeK) .* sqrt(PropDist.phi) ...
              + [model.tildephi_h, model.tildephi_delta];
newtildePhi(newtildePhi < -10) = -10;
newtildePhi(newtildePhi > 10) = 10;

newmodel = model;
newmodel.tildephi_h = newtildePhi(1:model.K);
newmodel.tildephi_delta = newtildePhi(model.K+1:end);

newmodel.phi_h = (exp(newmodel.tildephi_h) - 1) ./ (exp(newmodel.tildephi_h) + 1);
newmodel.phi_delta = (exp(newmodel.tildephi_delta) - 1) ./ (exp(newmodel.tildephi_delta) + 1);

numBlocks = newmodel.K + newmodel.tildeK;
nnz_total = numBlocks * (3 * newmodel.T - 2);

nonZeros = zeros(1, nnz_total);
idx = 1;

for i = 1:newmodel.K
    d = ((1 + newmodel.phi_h(i)^2) / newmodel.sigma2_h(i)) * ones(1, newmodel.T);
    d(1) = 1 / newmodel.sigma2_h(i);
    d(end) = 1 / newmodel.sigma2_h(i);

    nonZeros(idx:idx+newmodel.T-1) = d;
    idx = idx + newmodel.T;

    offd = -newmodel.phi_h(i) / newmodel.sigma2_h(i);
    len = 2 * newmodel.T - 2;
    nonZeros(idx:idx+len-1) = offd;
    idx = idx + len;
end

for j = 1:newmodel.tildeK
    d = ((1 + newmodel.phi_delta(j)^2) / newmodel.sigma2_delta(j)) * ones(1, newmodel.T);
    d(1) = 1 / newmodel.sigma2_delta(j);
    d(end) = 1 / newmodel.sigma2_delta(j);

    nonZeros(idx:idx+newmodel.T-1) = d;
    idx = idx + newmodel.T;

    offd = -newmodel.phi_delta(j) / newmodel.sigma2_delta(j);
    len = 2 * newmodel.T - 2;
    nonZeros(idx:idx+len-1) = offd;
    idx = idx + len;
end

Wtmp = sparse(iAll, jAll, nonZeros);

Lfreetmp = jitterChol(Wtmp);
ok = Lfreetmp * (F - mu);
newLogLphi = -0.5 * (newmodel.T * (newmodel.K + newmodel.tildeK)) * log(2*pi) ...
             + sum(log(diag(Lfreetmp))) ...
             - 0.5 * (ok' * ok);

if strcmp(model.exchangeablePriorphi, 'yes')
    newPriorphi_h = logmarginalizedNormalGam(newmodel.tildephi_h, ...
        model.priorPhi_h.mu0, model.priorPhi_h.k0, ...
        model.priorPhi_h.alpha0, model.priorPhi_h.beta0);

    newPriorphi_delta = logmarginalizedNormalGam(newmodel.tildephi_delta, ...
        model.priorPhi_delta.mu0, model.priorPhi_delta.k0, ...
        model.priorPhi_delta.alpha0, model.priorPhi_delta.beta0);
else
    newPriorphi_h = logNormal(newmodel.tildephi_h, ...
        model.priorPhi_h.mu0, model.priorPhi_h.s2);

    newPriorphi_delta = logNormal(newmodel.tildephi_delta, ...
        model.priorPhi_delta.mu0, model.priorPhi_delta.s2);
end

newPriorphi = newPriorphi_h + newPriorphi_delta;

[accept, uprob] = metropolisHastings(newLogLphi + newPriorphi, oldLogLphi + oldPriorphi, 0, 0);

if accept == 1
    model = newmodel;
    oldLogLphi = newLogLphi;
    oldPriorphi = newPriorphi;

    Lfree = Lfreetmp;
    Wonly = Wtmp;
    W = Wtmp + diag(sparse((1 ./ model.auxLikVar) * ones(1, n)));
    L = jitterChol(W)';
end

% ---------------------------------
% STORE OUTER VARIABLES
% ---------------------------------
samples.outer.h_0 = model.h_0;
samples.outer.delta_0 = model.delta_0;
samples.outer.sigma2_h = model.sigma2_h;
samples.outer.sigma2_delta = model.sigma2_delta;
samples.outer.tildephi_h = model.tildephi_h;
samples.outer.tildephi_delta = model.tildephi_delta;
samples.outer.phi_h = model.phi_h;
samples.outer.phi_delta = model.phi_delta;
% ---------------------------------
% ACCEPTANCE RATES (INNER BLOCKS ONLY)
% ---------------------------------
accRates.F = 100 * (acceptF / num_inner_total);

if strcmp(model.sampleFactorsByGibbs, 'yes') == 1
    accRates.Ft = 100 * ones(model.T, 1);
else
    accRates.Ft = 100 * (acceptFt / num_inner_total);
end

end