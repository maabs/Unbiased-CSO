clear all;
close all;
%randn('seed',0);
%rand('seed',0);
outdir = 'diagrams/';
rng('default')
%%
load tadawul_top250_daily_adj_close_and_rets.mat;
%%
data_d=data_d(10:30,1:end);
%time_d=time_d(:,1270:2000);
adj_d=adj_d(10:30,1:end);
time_d(1)
time_d(end)
%%
model.horizon  = 2; %forecast horizon
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

% rng(randi(10000),'twister');

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
%%

% SA OPTIONS (UNBIASED VERSION)
% ============================================================
model.horizon =5;
saOpts = struct();
% SA parameters
saOpts.Ksa       = 200;
saOpts.gamma     = 20;
saOpts.blockSize = 512;
saOpts.a0        = 3000/saOpts.gamma;
saOpts.aPow      = 0.6;
saOpts.xiClip    = 20;
saOpts.seed      = randi(1000);
% Unbiased estimator parameters
saOpts.B0        = 20;
saOpts.Lmax      = 6;
saOpts.M         = 100;
q = 4;
Lmax = saOpts.Lmax;
levels = 0:Lmax;
weights = (((levels + q)) .* (log(levels + q)).^2) ./ (2.^levels);
level_probs = weights / sum(weights);
saOpts.level_probs = level_probs;
% Forecast parameters
saOpts.localHorizon = 5;
saOpts.nForecastPerInner = 2;
% Misc
saOpts.threadCount = 1;
saOpts.verbose     = true;
%initialization
%saOpts.xi0 = out.xi_final;
saOpts.xi0 = zeros(model.N,1);   % safer for first test
%% ============================================================
% RUN UNBIASED SA
% ============================================================
out = unbiased_sa_portfolio_msv(adaptBundle, Langevin, saOpts);
%%
fig_path = ['/Users/alvarem/Library/CloudStorage/GoogleDrive-', ...
    'miguelangel.alvarezballesteros@kaust.edu.sa/Other computers/', ...
    'My MacBook Pro/MEGA/0KAUST/0CSO_project/Figs'];
op=out.obj_proxy_hist;
f=figure;
plot(op,  'LineWidth', 1.5); hold on; grid on;
%set(gca,'XScale','log')
ax = gca;  % get current axes
ax.FontSize = 15;
xlabel('SA iteration');
ylabel('$\bar{\mathrm{E}}[f(Z,\omega(\xi))]$', 'Interpreter', 'latex');
%title(sprintf('Evolution of the obj function in terms of the days'));
%legend('MSV betas','Uniform 1/N','Location','best');
out_file_t = fullfile(fig_path, ...
    sprintf('obj_funct_blocks_tawadul_first_daily_adj_close_and_rets_top20.pdf'));
%exportgraphics(f, out_file_t, 'ContentType','vector');
%%
t = 200;                    % pick time
w = out.beta_hist(:, t);   % N x 1
N = numel(w);
i = 1:N;
figure;
stem(i, w, 'filled'); hold on; grid on;
% Uniform weight line
yline(1/N, 'r--', 'LineWidth', 2);
xlabel('Asset index i');
ylabel('\beta_{i,t}');
title(sprintf('Weights at time t = %d', t));
legend('\beta_{i,t}', 'Uniform weight 1/N', 'Location','best');
%%
figure;
imagesc(out.beta_hist);
colorbar;
xlabel('Time t');
ylabel('Asset index i');
title('\beta_{i,t} heatmap');
colormap turbo;

