clear all;
close all;
outdir = 'diagrams/';
rng('default')
%%
load stoxx600_daily_adj_close_and_rets_top50.mat;

%%
model.horizon  = 1; %forecast horizon
Y =(data_d(:,1:end-model.horizon));
model.Y=Y;
model.actual = (data_d(:,end-model.horizon+1:end));
[N,T]=size(Y);
K=5;
%Y=data_d;
%if forecasting
% USER defined options
%
% -- Factor model or just simple MSV -- 
%    yes: factor MSV
%    no:  simple MSV  
model.useFactorModel = 'yes'; 
%
% -- Sample factors by Gibbs or not --
%    (this option has *no effect* if model.useFactorModel = 'no')
%      yes: it samples the factors by Gibbs (expensive, but exact) 
%      no:  it samples the factors by auxiliary Langevin (faster)
model.sampleFactorsByGibbs = 'no'; 
%
% -- Diagonal Sigmat matrix or not (i.e. indepedent factors or not) --
%      yes: the Sigmat matrices are all diagonal (angles are zero) 
%      no:  the Sigmat matrices have free form (angles are inferred)
model.diagonalSigmat = 'no';
%
% -- Exchangeable prior for the phis or just simple independent Gaussian with very
%    large variance
%    yes: exchangeable with normal-inverse gamma hyerprior 
%    no:  just a simple broad Gaussian 
model.exchangeablePriorphi = 'no';

if strcmp(model.useFactorModel, 'no') 
    K = N; 
end    

% create the Givens set
Givset = [];  % the indices 
for i=1:K
   for j=i+1:K
       Givset = [Givset; i j];
   end
end
tildeK = size(Givset,1);

Ytrue = Y;

% Add some missing values
%probNan = 0.1;  
%for t=1:T
%    r = rand(N,1);
%    r = find(r<=probNan); 
%    Y(r, t) = NaN; 
%end

% START CREATING THE MODEL STRUCTURE
model.N = N;
model.T = T;
model.K = K; 
model.Givset = Givset;
model.tildeK = size(Givset,1);

% PARAMETER INITIALIZATION FOR THE MCMC
model.deltas = zeros(model.tildeK, T); 
model.omegas = (0.5*pi)*( (exp(model.deltas)-1)./(exp(model.deltas) + 1));
model.hs = repmat(zeros(K,1), 1, T);
model.lambdas = exp(model.hs);

L = ones(N,K);
L(1:K,1:K) = triu(ones(K,K))';

% HYPERPARAMETER INITIALIZATION FOR MCMC
ind =  ~isnan(model.Y(:)); 
model.sigma2 = 0.01*var(model.Y(ind)); 
model.L = L;
model.Weights = randn(N,K);
model.sigma2weights = 2;
model.Ft = zeros(model.K, model.T);
size(model.L)

%%
%model.FFt = Ft;
model.phi_h = zeros(1 ,K);
model.tildephi_h = log((1 + model.phi_h)./(1 - model.phi_h));
model.h_0 = zeros(1 ,K);
model.sigma2_h = ones(1, K);
model.phi_delta = zeros(1, tildeK);
model.tildephi_delta = log((1 + model.phi_delta)./(1 - model.phi_delta));
model.delta_0 = zeros(1, tildeK);
model.sigma2_delta = ones(1, tildeK); 

% PRIOR OVER PHIS 
if strcmp(model.exchangeablePriorphi, 'yes') 
model.priorPhi_h.type = 'logmarginalizedNormalGam'; 
model.priorPhi_h.mu0 = 0;
model.priorPhi_h.k0 = 1;
model.priorPhi_h.alpha0 = 1;
model.priorPhi_h.beta0 = 1; 
model.priorPhi_delta.type = 'logmarginalizedNormalGam'; 
model.priorPhi_delta.mu0 = 0;
model.priorPhi_delta.k0 = 1;
model.priorPhi_delta.alpha0 = 1;
model.priorPhi_delta.beta0 = 1; 
else
model.priorPhi_h.type = 'logNormal'; 
model.priorPhi_h.mu0 = 0;
model.priorPhi_h.s2 = 100;
model.priorPhi_delta.type = 'logNormal'; 
model.priorPhi_delta.mu0 = 0;
model.priorPhi_delta.s2 = 100;
end
model.priorSigma2_h.sigmar = 5; 
model.priorSigma2_h.Ssigma = 0.01*model.priorSigma2_h.sigmar;  
model.priorSigma2_delta.sigmar = 5;
model.priorSigma2_delta.Ssigma = 0.01*model.priorSigma2_delta.sigmar;  

% INVERSE GAMMA PRIOR OVER THE LIKELIHOOD NOISE VARIANCE
model.priorSigma2.type = 'invgamma';  
model.priorSigma2.alpha0 = 0.001;
model.priorSigma2.beta0 = 0.001;
%%
% MCMC OPTIONS FOR BURNIN AND SAMPLING PHASES
mcmcoptions.adapt.T = 10;
mcmcoptions.adapt.Burnin = 0;
mcmcoptions.adapt.StoreEvery = 1;
mcmcoptions.adapt.disp = 1;
mcmcoptions.adapt.minAdapIters =10;
mcmcoptions.train.T = 10;
mcmcoptions.train.Burnin = 0;
mcmcoptions.train.StoreEvery = 2;
%%
sanityCheckMSV('preMCMC', model, [], mcmcoptions);
%%
% HERE WE RUN THE MCMC ALGORITHM FIRST TO ADAPT THE PROPOSAL AND THEN TO
% COLLECT THE SAMPLES

%rng(randi(10000),'twister');
Langevin = 1;
tic;
[model PropDist samples accRates] = mcmcAdapt(model, mcmcoptions.adapt, Langevin);
% training/sample collection phase
elapsedAdapt=toc;

%%
adaptBundle = struct();
adaptBundle.model = model;         % IMPORTANT: adapted model
adaptBundle.PropDist = PropDist;
adaptBundle.accRatesAdapt = accRates;
adaptBundle_base = adaptBundle;

%% ============================================================
% TEST FOR unbiased_sa_portfolio_msv_online.m
% Compatible with the setup style of test_mcmcTrain_blocks.m
%
% Assumes already in workspace:
%   model, mcmcoptions, Langevin, data_d
% ============================================================
rng(123,'twister');
%% ------------------------------------------------------------
% Choose an online split
% ------------------------------------------------------------
Yfull = data_d;
[Nfull, Tfull] = size(Yfull);

T0_online = Tfull-50;                 % initial training length
Honline   = Tfull - T0_online;   % online horizon

assert(Honline > 0, 'Need Tfull > T0_online');

%% ------------------------------------------------------------
% Build initial model for adaptation on the initial training window only
% ------------------------------------------------------------
model0 = model;
model0.Y = Yfull(:, 1:T0_online);
model0.N = Nfull;
model0.T = T0_online;
model0.horizon = Honline;

% Make sure time-varying fields are trimmed / resized consistently
tvFields = {'Ft','hs','deltas','omegas','lambdas'};
for ii = 1:numel(tvFields)
    f = tvFields{ii};
    if isfield(model0,f) && ~isempty(model0.(f))
        A = model0.(f);
        if size(A,2) >= T0_online
            model0.(f) = A(:,1:T0_online);
        else
            model0.(f) = [A, repmat(A(:,end),1,T0_online-size(A,2))];
        end
    end
end

if isfield(model0,'deltaFactors') && ~isempty(model0.deltaFactors)
    dF = model0.deltaFactors(:);
    if numel(dF) >= T0_online
        model0.deltaFactors = dF(1:T0_online);
    else
        model0.deltaFactors = [dF; repmat(dF(end), T0_online-numel(dF), 1)];
    end
end

%% ------------------------------------------------------------
% Adapt once on the initial training window
% ------------------------------------------------------------
Langevin=1;
%tic;
%[model_adapt, PropDist, samples_adapt, accRates] = ...
%    mcmcAdapt(model0, mcmcoptions.adapt, Langevin);
%toc;
model_adapt=model;
samples_adapt=samples;
%% ------------------------------------------------------------
% Build a FULL model object for the online wrapper
% Proposal parameters stay fixed; model state can evolve
% ------------------------------------------------------------
modelFull = model_adapt;
modelFull.Y = Yfull;
modelFull.N = Nfull;
modelFull.T = Tfull;
modelFull.horizon = Honline;

% Expand time-varying state fields to full panel length
for ii = 1:numel(tvFields)
    f = tvFields{ii};
    if isfield(modelFull,f) && ~isempty(modelFull.(f))
        A = modelFull.(f);
        if size(A,2) < Tfull
            modelFull.(f) = [A, repmat(A(:,end),1,Tfull-size(A,2))];
        elseif size(A,2) > Tfull
            modelFull.(f) = A(:,1:Tfull);
        end
    end
end

% Expand deltaFactors if needed
if isfield(modelFull,'deltaFactors') && ~isempty(modelFull.deltaFactors)
    dF = modelFull.deltaFactors(:);
    if numel(dF) < Tfull
        modelFull.deltaFactors = [dF; repmat(dF(end), Tfull-numel(dF), 1)];
    elseif numel(dF) > Tfull
        modelFull.deltaFactors = dF(1:Tfull);
    end
end
%% ------------------------------------------------------------
% onlineOpts
% ------------------------------------------------------------
onlineOpts = struct();
onlineOpts.T0 = T0_online;
onlineOpts.H  = Honline;

% number of SA updates performed at each rebalance date
onlineOpts.saInnerIters = 10;

% rebalance every localHorizon dates
onlineOpts.rebalanceEvery = 10;

% do not force model.horizon unless you know you need it
onlineOpts.forceModelHorizon = false;

onlineOpts.reseedPerRebalance = true;
onlineOpts.verbose = true;

%% ------------------------------------------------------------
% Base SA options passed into unbiased_sa_portfolio_msv at each rebalance
% ------------------------------------------------------------
saOptsBase = struct();
% SA parameters
saOptsBase.gamma     = 20;
saOptsBase.blockSize = 128;
saOptsBase.a0        = 40000/saOptsBase.gamma;
saOptsBase.aPow      = 0.6;
saOptsBase.seed      = 123;
saOptsBase.verbose   = true;
saOptsBase.xiClip    = 20;
saOptsBase.threadCount = 1;
% Truncated unbiased-style estimator parameters
saOptsBase.B0   = 5;
saOptsBase.Lmax = 6;
saOptsBase.M    = 10;
q = 6;
levels = 0:saOptsBase.Lmax;
weights = ((levels + q) .* (log(levels + q)).^2) ./ (2.^levels);
saOptsBase.level_probs = weights / sum(weights);

% Local forecast horizon used INSIDE each local unbiased gradient estimator
saOptsBase.localHorizon = onlineOpts.rebalanceEvery;
saOptsBase.nForecastPerInner = 100;
% Warm start xi
saOptsBase.xi0 = zeros(modelFull.N,1);
onlineOpts.saOptsBase = saOptsBase;
%% ------------------------------------------------------------
% Run the unbiased online wrapper
% ------------------------------------------------------------
tic;
[outOnline, adaptBundle] = unbiased_sa_portfolio_msv_online( ...
    adaptBundle, mcmcoptions, Langevin, onlineOpts);
toc;
%% ------------------------------------------------------------
% Save reproducibility file
% ------------------------------------------------------------

%out_dir = fullfile(pwd, 'repro_runs');
%if ~exist(out_dir, 'dir')
%    mkdir(out_dir);
%end
%
%runTag = datestr(now, 'yyyymmdd_HHMMSS');
%
%fname = sprintf(['repro_online_unbiased_stoxx600_N%d_T0%d_H%d_' ...
%                 'gamma%d_B0%d_Lmax%d_M%d_inner%d_reb%d_%s.mat'], ...Jo
%                 modelFull.N, ...
%                 onlineOpts.T0, ...
%                 onlineOpts.H, ...
%                 saOptsBase.gamma, ...
%                 saOptsBase.B0, ...
%                 saOptsBase.Lmax, ...
%                 saOptsBase.M, ...
%                 onlineOpts.saInnerIters, ...
%                 onlineOpts.rebalanceEvery, ...
%                 runTag);
%
%save_path = fullfile(out_dir, fname);
%
% Store RNG state at the end of the run
%rng_state_after = rng;
%
%% Store useful reproducibility metadata
%repro = struct();
%
%repro.description = 'Unbiased online SA portfolio MSV run';
%repro.created_at = datestr(now);
%repro.data_file = 'stoxx600_daily_adj_close_and_rets_top400.mat';
%
%repro.Nfull = Nfull;
%repro.Tfull = Tfull;
%repro.T0_online = T0_online;
%repro.Honline = Honline;
%
%repro.data_window.asset_idx = 1:Nfull;
%repro.data_window.time_idx = 1:Tfull;
%
%repro.rng_initial_seed = 123;
%repro.rng_state_after = rng_state_after;
%
%repro.onlineOpts = onlineOpts;
%repro.saOptsBase = saOptsBase;
%repro.mcmcoptions = mcmcoptions;
%repro.Langevin = Langevin;

% Save the important objects
%%save(save_path, ...
 %   'outOnline', ...
 %   'adaptBundle', ...
 %   'modelFull', ...
 %   'model0', ...
 %   'model_adapt', ...
 %   'samples_adapt', ...
 %   'PropDist', ...
 %   'accRates', ...
 %   'onlineOpts', ...
 %   'saOptsBase', ...
 %   'mcmcoptions', ...
 %   'Langevin', ...
 %   'repro', ...
 %   '-v7.3');

%fprintf('\nSaved reproducibility file:\n%s\n', save_path);


%% ------------------------------------------------------------
% Basic checks
% ------------------------------------------------------------
beta_star = outOnline.beta(:, end);
xi_star   = outOnline.xi(:, end);
disp('sum(beta_star):');
disp(sum(beta_star));

disp('[min(beta_star), max(beta_star)]:');
disp([min(beta_star), max(beta_star)]);

%% ------------------------------------------------------------
% Objective proxy
% ------------------------------------------------------------
figure;
plot(outOnline.obj_proxy, 'LineWidth', 1.5);
grid on;
xlabel('Online date h');
ylabel('Objective proxy');
title('Unbiased online SA objective proxy');

%% ------------------------------------------------------------
% Active assets
% ------------------------------------------------------------
%[row, column] = find(outOnline.beta > 0.01);
%disp('Indices of active weights > 0.01:');
%disp([row, column]);

%% ------------------------------------------------------------
% Heatmap of weights
% ------------------------------------------------------------
figure;
imagesc(outOnline.beta);
colorbar;
xlabel('Online date h');
ylabel('Asset index i');
title('\beta_{i,h} heatmap (unbiased online)');
colormap turbo;
%% ------------------------------------------------------------
% Snapshot of weights at one online date
% ------------------------------------------------------------
t = min(50, size(outOnline.beta,2));    % pick one online date
w = outOnline.beta(:, t);

N = numel(w);
i = 1:N;

figure;
stem(i, w, 'filled'); hold on; grid on;
yline(1/N, 'r--', 'LineWidth', 2);
xlabel('Asset index i');
ylabel('\beta_{i,t}');
title(sprintf('Weights at online date t = %d', t));
legend('\beta_{i,t}', 'Uniform weight 1/N', 'Location','best');

%% ------------------------------------------------------------
% Rebalance diagnostics
% ------------------------------------------------------------
figure;
stairs(outOnline.rebalanceFlag, 'LineWidth', 1.5);
grid on;
xlabel('Online date h');
ylabel('Rebalance flag');
title('Rebalance dates');

if isfield(outOnline, 'selectedLevel')
    figure;
    plot(outOnline.selectedLevel, 'o-');
    grid on;
    xlabel('Online date h');
    ylabel('Selected propagated level');
    title('Level used for propagated chain state');
end

%% ------------------------------------------------------------
% Wealth backtest over the online horizon
% ------------------------------------------------------------
% online window = last Honline dates after the initial training window
Y_trade = Yfull(:, T0_online+1 : T0_online+Honline);   % [N x H]
R = exp(Y_trade) - 1;                                   % simple returns

beta = outOnline.beta;                                  % [N x H]
V0 = 1;

V_msv  = zeros(1, Honline+1);
V_unif = zeros(1, Honline+1);
V_msv(1)  = V0;
V_unif(1) = V0;

beta_unif_full = ones(Nfull,1)/Nfull;

for t = 1:Honline
    rt = R(:,t);

    % ---------- MSV wealth ----------
    bt = beta(:,t);
    ok_msv = isfinite(rt) & isfinite(bt);

    if any(ok_msv)
        bt_ok = bt(ok_msv);
        s = sum(bt_ok);
        if s > 0
            bt_ok = bt_ok / s;
            r_msv = bt_ok' * rt(ok_msv);
        else
            r_msv = 0;
        end
    else
        r_msv = 0;
    end

    % ---------- Uniform wealth ----------
    ok_u = isfinite(rt);
    if any(ok_u)
        bu_ok = beta_unif_full(ok_u);
        bu_ok = bu_ok / sum(bu_ok);
        r_unif = bu_ok' * rt(ok_u);
    else
        r_unif = 0;
    end

    V_msv(t+1)  = V_msv(t)  * (1 + r_msv);
    V_unif(t+1) = V_unif(t) * (1 + r_unif);
end

tgrid = 0:Honline;

figure;
plot(tgrid, V_msv,  'LineWidth', 2.0); hold on; grid on;
plot(tgrid, V_unif, 'LineWidth', 2.0);
xlabel('t (days)');
ylabel('Wealth V_t');
title(sprintf('Wealth over online horizon H=%d', Honline));
legend('Unbiased MSV betas','Uniform 1/N','Location','best');

figure;
plot(tgrid, log(V_msv),  'LineWidth', 2.0); hold on; grid on;
plot(tgrid, log(V_unif), 'LineWidth', 2.0);
xlabel('t (days)');
ylabel('log Wealth');
title(sprintf('Log-wealth over online horizon H=%d', Honline));
legend('Unbiased MSV betas','Uniform 1/N','Location','best');

%% ------------------------------------------------------------
% Sharpe and simple performance table
% ------------------------------------------------------------
rp_msv  = zeros(Honline,1);
rp_unif = zeros(Honline,1);
turnover = zeros(Honline,1);

for t = 1:Honline
    rt = R(:,t);
    bt = beta(:,t);

    ok = isfinite(rt) & isfinite(bt);

    if any(ok)
        bt_ok = bt(ok);
        bt_ok = bt_ok / sum(bt_ok);
        rp_msv(t) = bt_ok' * rt(ok);

        bu_ok = beta_unif_full(ok);
        bu_ok = bu_ok / sum(bu_ok);
        rp_unif(t) = bu_ok' * rt(ok);
    else
        rp_msv(t) = 0;
        rp_unif(t) = 0;
    end

    if t > 1
        turnover(t) = sum(abs(beta(:,t) - beta(:,t-1)));
    end
end

W_msv  = cumprod(1 + rp_msv);
W_unif = cumprod(1 + rp_unif);

S_msv  = W_msv(end);
S_unif = W_unif(end);

APR_msv  = S_msv^(252/Honline) - 1;
APR_unif = S_unif^(252/Honline) - 1;

Vol_msv  = std(rp_msv)  * sqrt(252);
Vol_unif = std(rp_unif) * sqrt(252);

Sharpe_msv  = mean(rp_msv)  / std(rp_msv)  * sqrt(252);
Sharpe_unif = mean(rp_unif) / std(rp_unif) * sqrt(252);

max_drawdown = @(W) max((cummax(W) - W) ./ cummax(W));

MDD_msv  = max_drawdown(W_msv);
MDD_unif = max_drawdown(W_unif);

Calmar_msv  = APR_msv  / MDD_msv;
Calmar_unif = APR_unif / MDD_unif;

AvgTurnover = mean(turnover(2:end));

tc = 0.001;
rp_msv_tc = rp_msv - tc * turnover;
W_msv_tc = cumprod(1 + rp_msv_tc);
S_msv_tc = W_msv_tc(end);

fprintf('\n================= PERFORMANCE TABLE =================\n');
fprintf('Strategy        | FinalW |   APR   |  Vol   | Sharpe |  MDD  | Calmar\n');
fprintf('----------------------------------------------------------------------\n');
fprintf('Unbiased MSV-SA | %6.2f | %6.2f%% | %6.2f%% | %6.2f | %6.2f%% | %6.2f\n', ...
    S_msv, 100*APR_msv, 100*Vol_msv, Sharpe_msv, 100*MDD_msv, Calmar_msv);

fprintf('Uniform 1/N     | %6.2f | %6.2f%% | %6.2f%% | %6.2f | %6.2f%% | %6.2f\n', ...
    S_unif, 100*APR_unif, 100*Vol_unif, Sharpe_unif, 100*MDD_unif, Calmar_unif);
fprintf('----------------------------------------------------------------------\n');
fprintf('Avg turnover (Unbiased MSV): %.4f\n', AvgTurnover);
fprintf('Final wealth with TC (Unbiased MSV): %.2f\n', S_msv_tc);
fprintf('=====================================================\n');

%%
%% ------------------------------------------------------------
% performance metrics
% ------------------------------------------------------------

metrics = struct();

% Helper functions
ann_return = @(rp) prod(1 + rp).^(252/numel(rp)) - 1;
ann_vol    = @(rp) std(rp) * sqrt(252);
sharpe     = @(rp) mean(rp) / std(rp) * sqrt(252);

wealth     = @(rp) cumprod(1 + rp);
mdd_fun    = @(W) max((cummax(W) - W) ./ cummax(W));

avg_gain   = @(rp) mean(rp(rp > 0));
avg_loss   = @(rp) mean(rp(rp < 0));
win_trades = @(rp) mean(rp > 0);

%% Unbiased MSV-SA metrics
W_msv = wealth(rp_msv);

metrics.msv.FinalW = W_msv(end);
metrics.msv.AnnR   = ann_return(rp_msv);
metrics.msv.AnnV   = ann_vol(rp_msv);
metrics.msv.Sharpe = sharpe(rp_msv);
metrics.msv.MDD    = mdd_fun(W_msv);
metrics.msv.Calmar = metrics.msv.AnnR / metrics.msv.MDD;

metrics.msv.Gain   = avg_gain(rp_msv);
metrics.msv.Loss   = avg_loss(rp_msv);
metrics.msv.WT     = win_trades(rp_msv);
metrics.msv.TO     = mean(turnover(2:end));

%% Uniform 1/N metrics
W_unif = wealth(rp_unif);

metrics.unif.FinalW = W_unif(end);
metrics.unif.AnnR   = ann_return(rp_unif);
metrics.unif.AnnV   = ann_vol(rp_unif);
metrics.unif.Sharpe = sharpe(rp_unif);
metrics.unif.MDD    = mdd_fun(W_unif);
metrics.unif.Calmar = metrics.unif.AnnR / metrics.unif.MDD;

metrics.unif.Gain   = avg_gain(rp_unif);
metrics.unif.Loss   = avg_loss(rp_unif);
metrics.unif.WT     = win_trades(rp_unif);
metrics.unif.TO     = 0;

%% ------------------------------------------------------------
% Print table
% ------------------------------------------------------------

fprintf('\n========================= PERFORMANCE TABLE =========================\n');
fprintf('Strategy        | FinalW | %% gain | %% loss |  MDD  | %% WT  |   TO   | Ann.R | Ann.V | Sharpe | Calmar\n');
fprintf('--------------------------------------------------------------------------------------------------------\n');

fprintf('Unbiased MSV-SA | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f | %6.4f | %6.2f | %6.2f | %6.2f | %6.2f\n', ...
    metrics.msv.FinalW, ...
    100*metrics.msv.Gain, ...
    100*metrics.msv.Loss, ...
    100*metrics.msv.MDD, ...
    100*metrics.msv.WT, ...
    metrics.msv.TO, ...
    100*metrics.msv.AnnR, ...
    100*metrics.msv.AnnV, ...
    metrics.msv.Sharpe, ...
    metrics.msv.Calmar);

fprintf('Uniform 1/N     | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f | %6.4f | %6.2f | %6.2f | %6.2f | %6.2f\n', ...
    metrics.unif.FinalW, ...
    100*metrics.unif.Gain, ...
    100*metrics.unif.Loss, ...
    100*metrics.unif.MDD, ...
    100*metrics.unif.WT, ...
    metrics.unif.TO, ...
    100*metrics.unif.AnnR, ...
    100*metrics.unif.AnnV, ...
    metrics.unif.Sharpe, ...
    metrics.unif.Calmar);

fprintf('--------------------------------------------------------------------------------------------------------\n');

%% ============================================================
