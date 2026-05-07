clear all;
close all;
randn('seed',0);
rand('seed',0);
outdir = 'diagrams/';
%%
% LOAD the data 
%load toy1.mat;
%size(Y)
%%
%N = 2;   % number of log return sequences
%T = 200; % number of time instances 
%K = 2;   % number of latent factors 
% END LOADING the data 

load tadawul_top50_daily_adj_close_and_rets.mat;

%%

%%
model.horizon  = 10; %forecast horizon
Y =(data_q(:,1:end-model.horizon));
model.Y=Y;
model.actual = (data_q(:,end-model.horizon+1:end));
[N,T]=size(Y);
K=10;
%Y=data_q;

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
mcmcoptions.adapt.T = 20;
mcmcoptions.adapt.Burnin = 0;
mcmcoptions.adapt.StoreEvery = 1;
mcmcoptions.adapt.disp = 1;
mcmcoptions.adapt.minAdapIters = 5;
mcmcoptions.train.T = 10;
mcmcoptions.train.Burnin = 0;
mcmcoptions.train.StoreEvery = 2;
%%
sanityCheckMSV('preMCMC', model, [], mcmcoptions);
%%
% HERE WE RUN THE MCMC ALGORITHM FIRST TO ADAPT THE PROPOSAL AND THEN TO
% COLLECT THE SAMPLES
Langevin = 1;
tic;
[model PropDist samples accRates] = mcmcAdapt(model, mcmcoptions.adapt, Langevin); 
% training/sample collection phase
elapsedAdapt=toc;


%%

adaptBundle = struct();
adaptBundle.model = model;         % IMPORTANT: adapted model (has auxLikVar etc.)
adaptBundle.PropDist = PropDist;
adaptBundle.accRatesAdapt = accRates;

%%

%%
mcmcoptions.train.T = 200;
mcmcoptions.train.Burnin = 0;
mcmcoptions.train.StoreEvery = 1;

% modelFull must already contain the FULL data panel in modelFull.Y (N x TT)
% and all the other MSV fields needed by mcmcTrain / forecasting / gradients.

onlineOpts = struct();
onlineOpts.T0           = size(data_q,2)-model.horizon;    % initial training length
onlineOpts.H            = model.horizon;      % number of online days to run
onlineOpts.saInnerIters = 15;       % SA updates per day
onlineOpts.forceHorizon1 = true;   % horizon=1 each day
onlineOpts.reseedPerDay  = true;   % reproducible day-to-day

% Base SA options passed into sa_portfolio_msv each day
saOptsBase = struct();
saOptsBase.gamma     = 1;
saOptsBase.blockSize = 128;
saOptsBase.a0        = 8000;
saOptsBase.aPow      = 0.6;
saOptsBase.seed      = 123;    % base seed; online wrapper offsets per day
saOptsBase.verbose   = true;
saOptsBase.xiClip=20;

% Warm start (optional)
model.Y=data_q;
saOptsBase.xi0 = zeros(model.N,1);  % or your previous xi

onlineOpts.saOptsBase = saOptsBase;

% Call with existing adaptBundle
[outOnline, adaptBundle] = sa_portfolio_msv_online( ...
    model, mcmcoptions, Langevin, onlineOpts, adaptBundle);

%%
[row,column]=find(outOnline.beta>0.07);
[row,column];
outOnline.beta(find(outOnline.beta>0.07))




%% ===============================
% Reproducibility pack: save state
% ===============================

repro = struct();

% ---- Core outputs you want to keep ----
repro.outOnline   = outOnline;
repro.adaptBundle = adaptBundle;

% ---- Inputs needed to rerun and/or recompute metrics ----
% Save the EXACT objects used in the online run
repro.model       = model;         % includes model.Y/data_q as you set it
repro.mcmcoptions = mcmcoptions;
repro.Langevin    = Langevin;      % if this is a struct/config
repro.onlineOpts  = onlineOpts;

% ---- Also store the raw panel & time index explicitly (robust) ----
% Useful if later you modify model.Y or reload model differently.
repro.data_q      = data_q;        % N x T log-returns
if exist('time_q','var')
    repro.time_q  = time_q;        % 1 x T (strings or datenums)
end

% ---- If you have prices too, store them (optional but helpful) ----
if exist('adj_close','var')
    repro.adj_close = adj_close;   % N x T prices aligned with data_q
end

% ---- Minimal metadata ----
repro.meta = struct();
repro.meta.created_at  = datetime('now');
repro.meta.matlab_ver  = version;
repro.meta.hostname    = char(java.net.InetAddress.getLocalHost.getHostName); %#ok<JAVNM>
repro.meta.horizon     = model.horizon;

% ---- Save to disk (v7.3 supports large arrays/structs) ----
out_dir = fullfile(pwd, 'repro_runs');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

fname = sprintf('repro_online_tadawul_t40_every1_%s.mat', datestr(now,'yyyymmdd_HHMMSS'));
%save(fullfile(out_dir, fname), 'repro', '-v7.3');

fprintf('\nSaved reproducibility pack:\n  %s\n', fullfile(out_dir, fname));

%%

%%
% Option A: missing-aware returns + renormalize weights on available assets

[N, T] = size(data_q);

H  = model.horizon;
T0 = T - H;

Y_trade = data_q(:, T0+1 : T0+H);   % [N x H] log-returns (may contain NaN)
R = exp(Y_trade) - 1;               % [N x H] simple returns (NaNs propagate)

beta = outOnline.beta;              % [N x H] target weights (assumed finite, but we guard)

V0 = 1;

V_msv  = zeros(1, H+1);
V_unif = zeros(1, H+1);
V_msv(1)  = V0;
V_unif(1) = V0;

beta_unif_full = ones(N,1)/N;

for t = 1:H
    rt = R(:,t);

    % ---------- MSV wealth (renormalize on available returns) ----------
    bt = beta(:,t);

    ok_msv = isfinite(rt) & isfinite(bt);
    if any(ok_msv)
        bt_ok = bt(ok_msv);
        s = sum(bt_ok);
        if s > 0
            bt_ok = bt_ok / s;                 % renormalize on tradable set
            r_msv = bt_ok' * rt(ok_msv);
        else
            r_msv = 0;                         % degenerate weights => flat
        end
    else
        r_msv = 0;                             % nothing tradable => flat
    end

    % ---------- Uniform wealth (renormalize on available returns) ----------
    ok_u = isfinite(rt);
    if any(ok_u)
        bu_ok = beta_unif_full(ok_u);
        bu_ok = bu_ok / sum(bu_ok);            % = 1/|A_t| each, but explicit
        r_unif = bu_ok' * rt(ok_u);
    else
        r_unif = 0;
    end

    % ---------- update wealth ----------
    V_msv(t+1)  = V_msv(t)  * (1 + r_msv);
    V_unif(t+1) = V_unif(t) * (1 + r_unif);
end
% --- Plot wealth paths (and optionally log-wealth), assuming V_msv, V_unif exist ---
tgrid = 0:H;

figure;
plot(tgrid, V_msv,  'LineWidth', 2.0); hold on; grid on;
plot(tgrid, V_unif, 'LineWidth', 2.0);
xlabel('t (days)');
ylabel('Wealth V_t');
title(sprintf('Wealth over trading horizon H=%d', H));
legend('MSV betas','Uniform 1/N','Location','best');

% Optional: log-wealth (more readable if growth is large)
figure;
plot(tgrid, log(V_msv),  'LineWidth', 2.0); hold on; grid on;
plot(tgrid, log(V_unif), 'LineWidth', 2.0);
xlabel('t (days)');
ylabel('log Wealth');
title(sprintf('Log-wealth over trading horizon H=%d', H));
legend('MSV betas','Uniform 1/N','Location','best');
% Optional: normalize both to start at 1 (they already do if V0=1)
% but if you ever change V0, this keeps comparison clean:
% figure;
% plot(tgrid, V_msv./V_msv(1),  'LineWidth', 2.0); hold on; grid on;
% plot(tgrid, V_unif./V_unif(1), 'LineWidth', 2.0);
% xlabel('t (days)'); ylabel('Normalized wealth');
% title('Normalized wealth'); legend('MSV','Uniform','Location','best');
%%
% =========================
% Sharpe over the H-day window (scale by sqrt(H), not sqrt(252))
% Requires: data_q (N×T log-returns), outOnline.beta (N×H), model.horizon = H
% Uses the same missing-aware renormalization logic as your wealth code.
% =========================

[N, T] = size(data_q);
H  = model.horizon;
T0 = T - H;

Y_trade = data_q(:, T0+1 : T0+H);   % [N x H] log-returns (may contain NaN)
R = exp(Y_trade) - 1;               % [N x H] simple returns (NaNs propagate)

beta = outOnline.beta;              % [N x H] target weights
beta_unif_full = ones(N,1)/N;

% Store daily portfolio returns (simple) for Sharpe
rp_msv  = zeros(H,1);
rp_unif = zeros(H,1);

for t = 1:H
    rt = R(:,t);

    % ---------- MSV daily return ----------
    bt = beta(:,t);
    ok_msv = isfinite(rt) & isfinite(bt);

    if any(ok_msv)
        bt_ok = bt(ok_msv);
        s = sum(bt_ok);
        if s > 0
            bt_ok = bt_ok / s;                 % renormalize weights
            rp_msv(t) = bt_ok' * rt(ok_msv);
        else
            rp_msv(t) = 0;
        end
    else
        rp_msv(t) = 0;
    end

    % ---------- Uniform daily return ----------
    ok_u = isfinite(rt);
    if any(ok_u)
        bu_ok = beta_unif_full(ok_u);
        bu_ok = bu_ok / sum(bu_ok);            % = 1/|A_t|
        rp_unif(t) = bu_ok' * rt(ok_u);
    else
        rp_unif(t) = 0;
    end
end

% Optional: if you prefer to ignore "no tradable assets" days instead of 0-return days,
% replace rp_* zeros with NaN on those days and compute nanmean/nanstd accordingly.

% Risk-free rate (per day) over this window: set to 0 unless you want to include it
rf = 0;

excess_msv  = rp_msv  - rf;
excess_unif = rp_unif - rf;

% Window Sharpe (scale by sqrt(H))
mu_msv  = mean(excess_msv);
sd_msv  = std(excess_msv, 0);   % sample std
mu_unif = mean(excess_unif);
sd_unif = std(excess_unif, 0);

Sharpe_msv  = sqrt(H) * (mu_msv  / sd_msv);
Sharpe_unif = sqrt(H) * (mu_unif / sd_unif);

fprintf('Sharpe over H=%d days (scaled by sqrt(H)):\n', H);
fprintf('  MSV   : %.4f   (mean=%.4e, std=%.4e)\n', Sharpe_msv,  mu_msv,  sd_msv);
fprintf('  Unif  : %.4f   (mean=%.4e, std=%.4e)\n', Sharpe_unif, mu_unif, sd_unif);

% If you want the realized total return over the window too:
V_H_msv  = prod(1 + rp_msv)  - 1;
V_H_unif = prod(1 + rp_unif) - 1;
fprintf('Total simple return over window:\n');
fprintf('  MSV   : %.4f\n', V_H_msv);
fprintf('  Unif  : %.4f\n', V_H_unif);
%%
%% ===============================
% 1) Prepare returns and weights
% ===============================

[N, T] = size(data_q(:,end-model.horizon+1:end));

% Convert log returns to simple returns
R = exp(data_q(:,end-model.horizon+1:end)) - 1;   % [N x T]

beta_msv = outOnline.beta;   % must be [N x T]
beta_unif = ones(N,1)/N;

rp_msv  = zeros(T,1);
rp_unif = zeros(T,1);
turnover = zeros(T,1);

for t = 1:T
    
    rt = R(:,t);
    bt = beta_msv(:,t);
    
    ok = isfinite(rt) & isfinite(bt);
    
    if any(ok)
        bt_ok = bt(ok);
        bt_ok = bt_ok / sum(bt_ok);
        rp_msv(t) = bt_ok' * rt(ok);
        
        bu_ok = beta_unif(ok);
        bu_ok = bu_ok / sum(bu_ok);
        rp_unif(t) = bu_ok' * rt(ok);
    else
        rp_msv(t) = 0;
        rp_unif(t) = 0;
    end
    
    % Turnover (skip first day)
    if t > 1
        turnover(t) = sum(abs(beta_msv(:,t) - beta_msv(:,t-1)));
    end
end

%% ===============================
% 2) Wealth paths
% ===============================

W_msv  = cumprod(1 + rp_msv);
W_unif = cumprod(1 + rp_unif);

S_msv  = W_msv(end);
S_unif = W_unif(end);

%% ===============================
% 3) Annualized return (APR)
% ===============================

APR_msv  = S_msv^(252/T) - 1;
APR_unif = S_unif^(252/T) - 1;

%% ===============================
% 4) Volatility
% ===============================

Vol_msv  = std(rp_msv)  * sqrt(252);
Vol_unif = std(rp_unif) * sqrt(252);

%% ===============================
% 5) Sharpe ratio
% ===============================

Sharpe_msv  = mean(rp_msv)  / std(rp_msv)  * sqrt(252);
Sharpe_unif = mean(rp_unif) / std(rp_unif) * sqrt(252);

%% ===============================
% 6) Maximum Drawdown
% ===============================

max_drawdown = @(W) max((cummax(W) - W) ./ cummax(W));

MDD_msv  = max_drawdown(W_msv);
MDD_unif = max_drawdown(W_unif);

%% ===============================
% 7) Calmar ratio
% ===============================

Calmar_msv  = APR_msv  / MDD_msv;
Calmar_unif = APR_unif / MDD_unif;

%% ===============================
% 8) Average Turnover
% ===============================

AvgTurnover = mean(turnover(2:end));

%% ===============================
% 9) Optional transaction costs
% ===============================

tc = 0.001;   % 10bps per unit turnover

rp_msv_tc = rp_msv - tc * turnover;
W_msv_tc = cumprod(1 + rp_msv_tc);
S_msv_tc = W_msv_tc(end);

%% ===============================
% 10) Display Table
% ===============================

fprintf('\n================= PERFORMANCE TABLE =================\n');
fprintf('Strategy     | FinalW |   APR   |  Vol   | Sharpe |  MDD  | Calmar\n');
fprintf('---------------------------------------------------------------------\n');
fprintf('MSV-SA       | %6.2f | %6.2f%% | %6.2f%% | %6.2f | %6.2f%% | %6.2f\n', ...
    S_msv, 100*APR_msv, 100*Vol_msv, Sharpe_msv, 100*MDD_msv, Calmar_msv);

fprintf('Uniform 1/N  | %6.2f | %6.2f%% | %6.2f%% | %6.2f | %6.2f%% | %6.2f\n', ...
    S_unif, 100*APR_unif, 100*Vol_unif, Sharpe_unif, 100*MDD_unif, Calmar_unif);

fprintf('---------------------------------------------------------------------\n');
fprintf('Avg daily turnover (MSV): %.4f\n', AvgTurnover);
fprintf('Final wealth with TC (MSV): %.2f\n', S_msv_tc);
fprintf('=====================================================\n');