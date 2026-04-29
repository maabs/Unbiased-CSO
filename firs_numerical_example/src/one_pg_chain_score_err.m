function [enorm, ecomp, tsec] = one_pg_chain_score_err( ...
    y,T,N,M,B, ...
    in_dist,in_pars,trans,tr_pars,g,g_pars, ...
    seed, ...
    cpf_choice,traj_mode,trans_logpdf, ...
    store_parts,store_anc,store_logw, ...
    N_init,init_mode, ...
    g_anal, ...
    S0)

    t0 = tic;

    % Run one PG chain of length B (plus initial value)
    out_pg = pgibbs_run_init_pfmean( ...
        y, T, N, M, B, ...
        in_dist, in_pars, ...
        trans, tr_pars, ...
        g, g_pars, ...
        seed, ...
        cpf_choice, ...
        traj_mode, trans_logpdf, ...
        store_parts, store_anc, store_logw, ...
        N_init, init_mode);

    % X_paths: d_x × T × M × (B+1)
    X_paths = out_pg.X_paths;

    % Build paths_sel to select all chains & all PG iterations except the initializer (column 1)
    [~,~,M_,B1] = size(X_paths); %#ok<ASGLU>
    paths_sel = true(M_, B1);
    paths_sel(:,1) = false;   % exclude initial path

    % Average score over all M*(B) paths
    S = score_gaussian_from_paths_vectorized( ...
            y, X_paths, tr_pars.theta, tr_pars.q, g_pars.R, S0, ...
            paths_sel, [], false);   % don't override exclude_first_col, we handle via paths_sel

    g_chain = S.avg_total;           % [dr; dtheta; dq; dS0] (4×1)
    diff    = g_chain - g_anal;

    enorm   = norm(diff,2);
    ecomp   = abs(diff).';
    tsec    = toc(t0);
end

%% 6) Sweep over B and K independent chains
tic;
for ib = 1:KB
    B = Bs(ib);
    for k = 1:Krep
        [err_norm(k,ib), err_comp(k,ib,:), time_pg(k,ib)] = ...
            one_pg_chain_score_err( ...
                y,T,N,M,B, ...
                in_dist,in_pars,trans,tr_pars,g,g_pars, ...
                seed0 + 10000*k + ib, ...
                cpf_choice,traj_mode,trans_logpdf, ...
                store_parts,store_anc,store_logw, ...
                N_init,init_mode, ...
                g_anal, ...
                S0);
    end
end
toc;

%% 7) Aggregate and plot: relative MSE of the score vs B

% Component-wise MSE (as before)
mse_comp  = squeeze( mean( err_comp.^2 , 1) );   % KB × 4

% ----- Component-wise relative MSE -----
% Avoid division by zero if a true component is exactly zero
g_anal_vec = g_anal(:).';
den_comp = g_anal_vec.^2;
den_comp(den_comp == 0) = NaN; % or small eps

rel_mse_comp = mse_comp ./ den_comp;   % KB × 4

% ----- Total relative MSE -----
mse_total = squeeze( mean( sum(err_comp.^2, 3) , 1) );   % KB × 1
den_total = sum(g_anal_vec.^2);
rel_mse_total = mse_total / den_total;

% ---- Plot component-wise relative MSE ----
comp_names = {'dr','d\theta','dq','dS_0'};

figure; 
for j = 1:4
    subplot(2,2,j);
    loglog(Bs, rel_mse_comp(:,j), 'o-','LineWidth',1.6); hold on; grid on;
    % Reference ~ C / B
    Cj = rel_mse_comp(1,j) * Bs(1);
    loglog(Bs, Cj ./ Bs, '--','LineWidth',1.2);
    xlabel('B (PG chain length)'); 
    ylabel(['relative MSE(',comp_names{j},')']);
    title(['Relative MSE: ', comp_names{j}]);
    legend('relMSE','C \times B^{-1}','Location','southwest');
end

sgtitle(sprintf('Relative score MSE vs PG chain length B  |  K=%d, N=%d, M=%d, T=%d', ...
                Krep, N, M, T));

% ---- Plot total relative MSE ----
figure;
loglog(Bs, rel_mse_total, 'o-','LineWidth',1.6); hold on; grid on;
Ctot = rel_mse_total(1) * Bs(1);
loglog(Bs, Ctot ./ Bs, '--','LineWidth',1.2);
xlabel('B'); ylabel('Total relative MSE');
title('Total relative score MSE vs PG chain length');
legend('relMSE','C \times B^{-1}','Location','southwest');
x%%
% comparison of the score functions 

%% ===== PG score vs analytical score in a 1D Gaussian SSM =====
clear; clc;

%% 1) Model & data
T     = 10;
theta = 0.95;
q     = 0.2^2;          % state variance
r     = 0.3^2;          % obs variance
S0    = q/(1-theta^2);  % initial variance (stationary) -- or any positive number

rng(10);
x = zeros(1,T); x(1) = sqrt(S0)*randn;
for t=2:T, x(t) = theta*x(t-1) + sqrt(q)*randn; end
y = x + sqrt(r)*randn(1,T);

%% 2) User functions for PF/CPF
in_pars.mu = 0; in_pars.Sigma = S0+0.2;
in_dist = @(p,N,M) reshape(p.mu + sqrt(p.Sigma)*randn(1,N*M), 1, N, M);

tr_pars.theta = theta; tr_pars.sig = sqrt(q);
trans = @(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

g_pars.R = r;
% Force 1×N×M shape (robust for M=1)
g = @(yt,Xt,p,t) reshape( -0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
                          1, size(Xt,2), size(Xt,3));

% Backward-simulation log transition (optional)
trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - theta*X_prev).^2)/q - 0.5*log(2*pi*q);   % 1×N

%% 3) Run Particle Gibbs (initializer = PF weighted means)
N = 2;                  % particles
M = 2;                    % chains in parallel
B = 256*16;                  % PG links (stored paths = B+1 incl. initializer)
seed0 = 123;

cpf_choice  = "cpf_parallel";   % or "cpf"
traj_mode   = "backward";       % "ancestors" also works
N_init      = N;
init_mode   = "weighted";       % "weighted" or "resampled"

store_parts = false; store_anc=false; store_logw=false;

out_pg = pgibbs_run_init_pfmean( ...
    y, T, N, M, B, ...
    in_dist, in_pars, ...
    trans, tr_pars, ...
    g, g_pars, ...
    seed0, ...
    cpf_choice, ...
    traj_mode, trans_logpdf, ...
    store_parts, store_anc, store_logw, ...
    N_init, init_mode);

% X_paths: d_x × T × M × (B+1)  (here d_x=1)
X_paths = out_pg.X_paths;

%% 4) Analytical score via smoother (Fisher identity in closed form)
[sc_anal, parts] = score_gaussian_ssm(y, theta, q, r, S0);

% Pack in vector (obs; trans; init) = [dr; dtheta; dq; dS0]
g_anal = [sc_anal.dr; sc_anal.dtheta; sc_anal.dq; sc_anal.dS0];

%% 5) Score from PG paths (Monte Carlo average over stored trajectories)
% Exclude the initializer (slot 1) to use only PG samples:
S_mc = score_gaussian_from_paths(y, X_paths, theta, q, r, S0, true);
g_mc = [S_mc.avg_obs; S_mc.avg_trans; S_mc.avg_init];   % [dr; dtheta; dq; dS0]

%% 6) Compare
names = {'dr','d\theta','dq','dS_0'}';
abs_err = abs(g_mc - g_anal);
rel_err = abs_err ./ max(1e-12, abs(g_anal));
fprintf('\nAnalytical score:   [dr dtheta dq dS0] = [%+.4e %+.4e %+.4e %+.4e]\n', g_anal);
fprintf('PG-path score:      [dr dtheta dq dS0] = [%+.4e %+.4e %+.4e %+.4e]\n', g_mc);
fprintf('Absolute error:                           [% .3e  % .3e  % .3e  % .3e]\n', abs_err);
fprintf('Relative error:                           [% .3e  % .3e  % .3e  % .3e]\n\n', rel_err);

figure;
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
bar([g_anal g_mc]); grid on;
title('Score components'); ylabel('value');
set(gca,'XTickLabel',names);
legend('Analytical','PG paths','Location','best');

nexttile;
bar([abs_err rel_err]); grid on;
title('Abs / Rel error');
set(gca,'XTickLabel',names);
legend('abs err','rel err','Location','best');


%%
% simulate (as in your earlier setup)
T=20; theta=0.95; q=0.2^2; r=0.3^2; S0=q/(1-theta^2);
rng(0); x = zeros(1,T); x(1)=sqrt(S0)*randn;
for t=2:T, x(t)=theta*x(t-1)+sqrt(q)*randn; end
y = x + sqrt(r)*randn(1,T);

% score
[sc, parts] = score_gaussian_ssm(y, theta, q, r, S0)

%% === Gaussian SSM, Particle Gibbs vs RTS smoother ===
clear; clc;

%% 1) Model params & data
T     = 20;                   % length
theta = 0.95                  % AR(1) coefficient
S_hm  = 0.2^2                 % state noise variance (q)
S_o   = 0.3^2                 % obs noise variance (r)
S_or  = S_hm/(1-theta^2)      % suggested stationary var for init (or use the S_or you want)

d_x = 1; d_y = 1;
rng(10);

x_true = zeros(1,T);
x_true(1) = sqrt(S_or)*randn;
for t=2:T
    x_true(t) = theta*x_true(t-1) + sqrt(S_hm)*randn;
end
y = x_true + sqrt(S_o)*randn(1,T);


plot(y)
hold on;
plot(x_true)
hold off;

%% 2) User functions for PF/CPF
% Initial sampler: returns d_x × N × M
in_pars.mu = 0; in_pars.Sigma = S_or;
in_dist = @(p,N,M) reshape(p.mu + sqrt(p.Sigma)*randn(1,N*M), 1, N, M);

% Transition sampler: X_t = theta * X_{t-1} + N(0, S_hm)
tr_pars.theta = theta; tr_pars.sig = sqrt(S_hm);
trans = @(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

% Observation log-likelihood: y_t | x_t ~ N(x_t, S_o)
g_pars.R = S_o;
g = @(yt,Xt,p,t) reshape( -0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), 1, size(Xt,2), size(Xt,3) );

% For backward simulation (optional trajectory mode)
trans_logpdf = @(x_next, X_prev, pars, t) ...        % returns 1×N
    -0.5*((x_next - theta*X_prev).^2)/S_hm - 0.5*log(2*pi*S_hm);

%% 3) RTS smoother (ground truth for linear Gaussian)
[m_f, P_f, m_s, P_s] = rts_smoother_1d(y, theta, S_hm, S_o, 1, 0, S_or); %#ok<ASGLU>
% m_s: 1×T, P_s: 1×T

%% 4) Particle Gibbs settings
N           = 20;         % particles
M           = 2;           % chains per iteration
B           = 256;         % PG links (you can vary)
seed0       = 123;
cpf_choice  = "cpf";   % or "cpf"
traj_mode   = "backward";       % "ancestors" also works
N_init      = N;                % PF size for initializer
init_mode   = "weighted";       % "weighted" or "resampled"
store_parts = false; store_anc = false; store_logw = false;

out_pg = pgibbs_run_init_pfmean( ...
    y, T, N, M, B, ...
    in_dist, in_pars, ...
    trans, tr_pars, ...
    g, g_pars, ...
    seed0, ...
    cpf_choice, ...
    traj_mode, trans_logpdf, ...
    store_parts, store_anc, store_logw, ...
    N_init, init_mode);

% out_pg.X_paths is d_x × T × M × (B+1)
% Take mean over chains and iterations (optionally exclude initializer at slot 1)
mean_pg_all = squeeze( mean( mean(out_pg.X_paths, 4), 3) );      % 1×T, using all B+1
% mean_pg = squeeze( mean( mean(out_pg.X_paths(:,:,:,2:end),4), 3) ); % exclude initializer

%% 5) Diagnostics & comparison
mse_pg_vs_rts = mean( (mean_pg_all - m_s).^2 );

fprintf('PG vs RTS (mean path MSE over time): %.4e\n', mse_pg_vs_rts);

% Pick a few times to look at cross-sections if you like
t_sel = [50 100 150 200];

%% 6) Plots
figure; 
subplot(2,1,1);
plot(1:T, x_true, '-', 'LineWidth',1); hold on; grid on;
plot(1:T, y, '.', 'MarkerSize',8);
plot(1:T, m_s, '-', 'LineWidth',1.8);
plot(1:T, mean_pg_all, '--', 'LineWidth',1.8);
legend('x true','y','RTS mean','PG mean','Location','best');
xlabel('t'); ylabel('state');
title(sprintf('Gaussian SSM: PG (%s/%s) vs RTS | N=%d, M=%d, B=%d', ...
    char(cpf_choice), char(traj_mode), N, M, B));

subplot(2,1,2);
plot(1:T, (mean_pg_all - m_s).^2, 'LineWidth',1.2); grid on;
xlabel('t'); ylabel('squared error'); title('PG mean vs RTS mean (per-time squared error)');



%% === Test: PG with init=PF-mean, MSE vs B (B = 2.^l), K replicates ===
% Requires: pgibbs_run_init_pfmean.m, pf_parallel.m, cpf_parallel.m and/or cpf.m, rts_smoother_1d.m

%% 1) Model + data (1D LGSSM)
T   = 10;   rho = 0.95;  q = 0.2^2;  r = 0.3^2;  H = 1;
P0  = q/(1 - rho^2);      m0 = 0;     d_x = 1;
seed_data = 101;
rng(seed_data);
x_true = zeros(1,T);  x_true(1) = m0 + sqrt(P0)*randn;
for t = 2:T, x_true(t) = rho*x_true(t-1) + sqrt(q)*randn; end
y = H*x_true + sqrt(r)*randn(1,T);

% User fns (vectorized over N, M)
in_pars.mu = m0; in_pars.Sigma = P0;
in_dist = @(p,N,M) reshape(p.mu + sqrt(p.Sigma)*randn(1,N*M), 1, N, M);

tr_pars.rho = rho; tr_pars.sig = sqrt(q);
trans = @(Xprev,p,t) p.rho*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

g_pars.H = H; g_pars.R = r;
g = @(yt,Xt,p,t) -0.5*((yt - p.H*Xt).^2)/p.R - 0.5*log(2*pi*p.R);

% Kalman smoother (reference)
[~, ~, m_s, ~] = rts_smoother_1d(y, rho, q, r, H, m0, P0);   % 1×T

%% 2) PG settings
N          = 20;              % particles per CPF
M          = 2;                % chains per iteration
seed0      = 123;              % base seed (we'll offset per replicate)

% Choose kernel implementation: "cpf" or "cpf_parallel"
cpf_choice = "cpf";

% Trajectory sampler: "ancestors" or "backward"
traj_mode  = "backward";       % switch to "ancestors" if desired
trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - pars.rho*X_prev).^2)/q - 0.5*log(2*pi*q);   % 1×N; only used for "backward"

% Return only trajectories (we don't need internals here)
store_particles = false; store_ancestors = false; store_logw = false;

%% 3) Grid over B and K replicates
l_vals = 2:9;                  % B in {4,8,16,...,512}
Bs     = 2.^l_vals;  KB = numel(Bs);
K      = 20;                   % independent replicates per B
N_init=N; init_mode="weighted";
% Per-time MSE vs B
mse_vs_B_T = zeros(KB, T);

for ib = 1:KB
    B = Bs(ib);
    sqerr_rep = zeros(K, T);               % K × T

    for k = 1:K
        seed_k = seed0 + 1e6*(k-1);

        out = pgibbs_run_init_pfmean( ...
            y, T, N, M, B, ...
            in_dist, in_pars, ...
            trans, tr_pars, ...
            g, g_pars, ...
            seed_k, ...
            cpf_choice, ...
            traj_mode, trans_logpdf, ...
            store_particles, store_ancestors, store_logw);

        % Mean over M chains and over the (B+1) stored paths (includes the PF-mean initializer)
        mean_pg = squeeze( mean( mean(out.X_paths, 4), 3) );   % 1×T

        % Per-time squared error vs RTS smoother
        sqerr_rep(k, :) = (mean_pg - m_s).^2;
    end

    % Observed MSE across K replicates (per time)
    mse_vs_B_T(ib, :) = mean(sqerr_rep, 1);
end

%% 4) Plot section — choose which times to display
t_sel = 5;    % e.g., scalar (50) or vector like [40 80 120]

% Reduce per-time MSE to one scalar per B by averaging over t_sel
mse_vs_B = mean(mse_vs_B_T(:, t_sel), 2);


figure;
loglog(Bs, mse_vs_B, 'o-', 'LineWidth',1.6, 'MarkerSize',6); hold on; grid on;
xlabel('B (CPF iterations)'); 
ylabel(sprintf('MSE at t=%s', mat2str(t_sel)));
title(sprintf('PG (init=PF mean) | kernel=%s | traj=%s | N=%d, M=%d, K=%d', ...
      cpf_choice, traj_mode, N, M, K));
% Reference slopes
C1 = mse_vs_B(end) * Bs(end);          % ~ C / B
loglog(Bs, C1 ./ Bs, '--');
C2 = mse_vs_B(end) * sqrt(Bs(end));    % ~ C / sqrt(B)
loglog(Bs, C2 ./ sqrt(Bs), ':');
legend('Observed MSE', 'C / B', 'C / sqrt(B)', 'Location','southwest');

% Slope fit on log–log (diagnostic)
p = polyfit(log(Bs(:)), log(mse_vs_B(:)), 1);
fprintf('Observed MSE slope (log–log) at t=%s: %.3f\n', mat2str(t_sel), p(1));

%%

%{
%% === CPF test: MSE vs B (B = 2.^l), per-time, with K replicates ===
% Requires: cpf_parallel.m (with traj_mode & trans_logpdf), rts_smoother_1d.m

%% 1) Model + data (1D LGSSM)
T   = 20;   rho = 0.95;  q = 0.2^2;  r = 0.3^2;  H = 1;
P0  = q/(1 - rho^2);  m0 = 0;  d_x = 1;

rng(1);
x_true = zeros(1,T);  x_true(1) = m0 + sqrt(P0)*randn;
for t = 2:T, x_true(t) = rho*x_true(t-1) + sqrt(q)*randn; end
y = H*x_true + sqrt(r)*randn(1,T);

% User fns (vectorized over N, M)
in_pars.mu = m0; in_pars.Sigma = P0;
in_dist = @(p,N,M) reshape(p.mu + sqrt(p.Sigma)*randn(1,N*M), 1, N, M);

tr_pars.rho = rho; tr_pars.sig = sqrt(q);
trans = @(Xprev,p,t) p.rho*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

g_pars.H = H; g_pars.R = r;
g = @(yt,Xt,p,t) -0.5*((yt - p.H*Xt).^2)/p.R - 0.5*log(2*pi*p.R);

% Kalman filter + RTS smoother (reference)
[~, ~, m_s, ~] = rts_smoother_1d(y, rho, q, r, H, m0, P0);   % smoothed mean 1×T

%% 2) CPF settings
N     = 10;          % particles per CPF
M     = 10;            % parallel chains per iteration (per replicate)
seed0 = 123;          % base seed
parallelization=false;
% Trajectory sampler: "ancestors" or "backward"
traj_mode = "ancestors";  % change to "backward" to use backward sampling

% Transition log-density (only used if traj_mode == "backward")
trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - pars.rho*X_prev).^2)/q - 0.5*log(2*pi*q);   % 1×N

% Storage return flags (we only need the trajectory for MSE)
store_particles = false; store_ancestors = false; store_logw = false;

% 3) Grid over B and K replicates
l_vals = 2:4;                % e.g. B in {4,8,...,1024}
Bs     = 2.^l_vals;  KB = numel(Bs);
K      = 20;                  % independent replicates per B

% Per-time MSE vs B
mse_vs_B_T = zeros(KB, T);
tic

for ib = 1:KB
    B = Bs(ib);
    sqerr_rep = zeros(K, T);          % K × T

    for k = 1:K
        seed_k = seed0 + 10^6*(k-1);

        % initial reference path (neutral start): use smoother mean replicated over M
        x_ref0 = repmat(reshape(m_s, d_x, T, 1), 1, 1, M);   % d_x×T×M

        % run CPF chain for B iterations
        % collect sampled paths: d_x×T×M×B (but we only need means over M,B)
        % to save memory, accumulate mean on the fly:
        mean_over_MB = zeros(1, T);   % running mean across M×B samples
        count = 0;

        % iter 1
        %{
        out = cpf_parallel( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, trans_pars, ...
        g, g_pars, ...
        x_ref, seed, ...
        traj_mode, trans_logpdf, ...
        store_particles, store_ancestors, store_logw)

        %}
        if parallelization
        out = cpf_parallel(y, T, N, M, in_dist, in_pars, trans, tr_pars, g, g_pars, ...
                           x_ref0, seed_k, traj_mode, trans_logpdf, ...
                           store_particles, store_ancestors, store_logw);
        else
        out = cpf(y, T, N, M, in_dist, in_pars, trans, tr_pars, g, g_pars, ...
                           x_ref0, seed_k, traj_mode, trans_logpdf, ...
                           store_particles, store_ancestors, store_logw);
        end
        X_samp = out.sampled_path;                 % 1×T×M
        mean_over_MB = mean_over_MB + squeeze(mean(X_samp, 3));  % 1×T
        count = count + 1;

        % iters 2..B
        for b = 2:B
            x_ref_b = X_samp;                      % condition on previous sampled paths per filter

            if parallelization
            out = cpf_parallel(y, T, N, M, in_dist, in_pars, trans, tr_pars, g, g_pars, ...
                               x_ref_b, seed_k + b - 1, traj_mode, trans_logpdf, ...
                               store_particles, store_ancestors, store_logw);
            else
                out = cpf(y, T, N, M, in_dist, in_pars, trans, tr_pars, g, g_pars, ...
                               x_ref_b, seed_k + b - 1, traj_mode, trans_logpdf, ...
                               store_particles, store_ancestors, store_logw);
            end
            
            X_samp = out.sampled_path;            % 1×T×M
            mean_over_MB = mean_over_MB + squeeze(mean(X_samp, 3)); % add mean over M
            count = count + 1;
        end

        mean_cpf = mean_over_MB / count;          % mean over all B (and over M)
        sqerr_rep(k, :) = (mean_cpf - m_s).^2;    % per-time squared error
    end

    mse_vs_B_T(ib, :) = mean(sqerr_rep, 1);       % MSE across K replicates (per time)
end
elapsed = toc;        % stop timer, returns time in seconds
fprintf("strategy is: %.i  ",parallelization)
fprintf('Elapsed time: %.4f seconds\n', elapsed);
%%

%% 4) Plot section — choose the time(s) here
t_sel=10;                  % e.g., scalar or vector like [40 80 120]

mse_vs_B = mean(mse_vs_B_T(:, t_sel), 2);   % average over chosen times → KB×1

figure; 
loglog(Bs, mse_vs_B, 'o-', 'LineWidth',1.6, 'MarkerSize',6); hold on; grid on;
xlabel('B (CPF iterations)'); ylabel(sprintf('MSE at t=%s', mat2str(t_sel)));
title(sprintf('CPF mean MSE vs B  (traj\\_mode=%s, N=%d, M=%d, K=%d)', ...
      traj_mode, N, M, K));

% reference slopes
C1 = mse_vs_B(end) * Bs(end);          % ~C/B
loglog(Bs, C1 ./ Bs, '--');
C2 = mse_vs_B(end) * sqrt(Bs(end));    % ~C/sqrt(B)
loglog(Bs, C2 ./ sqrt(Bs), ':');
legend('Observed MSE', 'C / B', 'C / sqrt(B)', 'Location','southwest');

p = polyfit(log(Bs(:)), log(mse_vs_B(:)), 1);
fprintf('Slope (log–log) at t=%s: %.3f\n', mat2str(t_sel), p(1));

%}

%%
% In the following "if" we test the conditional particle filter, we 
% see that the mse follows the expected rate proportional to 1/B where
% B is the chain lenght.
if True
%% ===== 1) Model + data (1D LGSSM) =====
T   = 20;
rho = 0.95; q = 0.2^2; r = 0.3^2; H = 1;
P0  = q/(1 - rho^2); m0 = 0;
d_x = 1;                       % <-- explicit: 1D state

rng(1);
x_true = zeros(1,T);
x_true(1) = m0 + sqrt(P0)*randn;
for t = 2:T
    x_true(t) = rho*x_true(t-1) + sqrt(q)*randn;
end
y = H*x_true + sqrt(r)*randn(1,T);

% User fns (vectorized over N,M)
in_pars.mu = m0; in_pars.Sigma = P0;
in_dist = @(p,N,M) reshape(p.mu + sqrt(p.Sigma)*randn(1,N*M), 1, N, M);

tr_pars.rho = rho; tr_pars.sig = sqrt(q);
trans = @(Xprev,p,t) p.rho*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

g_pars.H = H; g_pars.R = r;
g = @(yt,Xt,p,t) -0.5*((yt - p.H*Xt).^2)/p.R - 0.5*log(2*pi*p.R);

%% ===== 2) CPF experiment params =====
N     = 100;        % particles per CPF
M     = 2;          % run M parallel CPF chains each iteration
B     = 1000;        % number of CPF iterations
seed0 = 123;        % base seed

% Reference path x_ref: d_x × T × M (replicate same path across M)
x_true_row = x_true(:).';                 % ensure 1×T
x_ref = repmat(reshape(x_true_row, d_x, T, 1), 1, 1, M);   % 1×T×M

% Storage for sampled paths per iteration & per filter
X_paths = zeros(d_x, T, M, B);            % 1 × T × M × B

%% ===== 3) Iteration 1: condition on x_ref, then store sampled paths =====
out1 = cpf_parallel(y, T, N, M, in_dist, in_pars, trans, tr_pars, g, g_pars, x_ref, seed0);
% sanity: sampled_path must be d_x×T×M
assert(isequal(size(out1.sampled_path), [d_x, T, M]), 'sampled_path has wrong size.');
X_paths(:,:,:,1) = out1.sampled_path;

%% ===== 4) Iterations 2..B: condition on previous sampled paths (per filter) =====
for b = 2:B
    x_ref = X_paths(:,:,:,b-1);                          % 1×T×M
    outb  = cpf_parallel(y, T, N, M, in_dist, in_pars, trans, tr_pars, g, g_pars, x_ref, seed0 + b - 1);
    assert(isequal(size(outb.sampled_path), [d_x, T, M]), 'sampled_path has wrong size (iter %d).', b);
    X_paths(:,:,:,b) = outb.sampled_path;
end

%% ===== 5) Kalman smoothing (RTS) for ground truth smoothing mean =====
[m_f, P_f, m_s, P_s] = rts_smoother_1d(y, rho, q, r, H, m0, P0);

%% 6) Compare: mean of sampled paths vs KF (filter) and RTS (smoother)
% Assumes you have: x_true (1×T), mean_cpf (1×T), and the RTS/KF function.
% If m_f / m_s aren't in scope yet, compute them now:
if ~(exist('m_f','var')==1 && exist('m_s','var')==1 && ~isempty(m_f) && ~isempty(m_s))
    [m_f, P_f, m_s, P_s] = rts_smoother_1d(y, rho, q, r, H, m0, P0);
else
    % If only m_f/P_f missing:
    if ~(exist('P_f','var')==1), [~, P_f, ~, ~] = rts_smoother_1d(y, rho, q, r, H, m0, P0); end
end

% RMSE vs smoother (as before)
err = mean_cpf - m_s;
rmse = sqrt(mean(err.^2));
fprintf('RMSE (CPF mean over M=%d, B=%d) vs RTS smoothed mean: %.4g\n', M, B, rmse);

% ---- Plot: true, Kalman Filter (filtered mean), RTS smoother, CPF mean ----
t = 1:T;

figure; hold on; grid on;
% Optional filtered uncertainty band (±2σ)
doBand = true;
if doBand
    sigma_f = sqrt(P_f);                 % 1×T
    fill([t, fliplr(t)], [m_f+2*sigma_f, fliplr(m_f-2*sigma_f)], ...
         [0.85 0.92 1.00], 'EdgeColor','none', 'FaceAlpha',0.4); % light blue band
end

plot(x_true, 'k-', 'LineWidth',1);                      % true state
plot(m_f,    'g-.', 'LineWidth',1.2);                   % Kalman filter (filtered mean)
plot(m_s,    'b-',  'LineWidth',1.6);                   % RTS smoother (smoothed mean)
plot(mean_cpf,'r--', 'LineWidth',1.4);                  % CPF mean across M,B
plot(y)                                                 % Observations

legend_entries = {'True x','KF (filtered mean)','RTS (smoothed mean)','CPF mean'};
if ~doBand
    legend(legend_entries, 'Location','best');
else
    legend(['KF ±2σ band', legend_entries], 'Location','best');
end

xlabel('time'); ylabel('state');
title(sprintf('KF & RTS vs CPF mean  (M=%d, B=%d, N=%d)', M, B, N));

%%

%% ===== Experiment: K independent replicates → per-time MSE vs B =====
% Assumes you already defined:
%   cpf_parallel, rts_smoother_1d, y,T,rho,q,r,H,m0,P0,d_x (=1), 
%   in_dist,in_pars,trans,tr_pars,g,g_pars

% Fixed CPF settings
N     = 20;           % particles per CPF
M     = 2;            % parallel chains per iteration
seed0 = 123;          % base seed for reproducibility
% Grid B = 2.^l
l_vals = 2:10;         % e.g., B in {4,8,16,32,64,128,256}
Bs     = 2.^l_vals;
KB     = numel(Bs);

% # independent replicates per B
K = 50;

% RTS smoother reference (smoothed mean)
[~, ~, m_s, ~] = rts_smoother_1d(y, rho, q, r, H, m0, P0)   % m_s: 1×T

% Storage: per-B, per-time MSE and MC baseline
mse_vs_B_T     = zeros(KB, T);    % observed MSE at each time t
mc_pred_vs_B_T = zeros(KB, T);    % MC variance /(M*B) at each t

for ib = 1:KB
    B = Bs(ib);

    % ---- accumulate K replicate errors per time ----
    % sqerr_rep: K × T (each row is replicate k's squared error at each t)
    sqerr_rep = zeros(K, T);

    for k = 1:K
        seed_k = seed0 + 10^6 * (k-1);

        % initial reference path (use RTS smoother mean, size d_x×T×M)
        x_ref0 = repmat(reshape(m_s, d_x, T, 1), 1, 1, M);

        % run CPF chain for B iterations, store samples
        X_paths = zeros(d_x, T, M, B);                 % 1×T×M×B
        out1 = cpf_parallel(y, T, N, M, ...
                            in_dist, in_pars, trans, tr_pars, g, g_pars, ...
                            x_ref0, seed_k);
        X_paths(:,:,:,1) = out1.sampled_path;
        for b = 2:B
            x_ref_b = X_paths(:,:,:,b-1);             % 1×T×M
            outb = cpf_parallel(y, T, N, M, ...
                                in_dist, in_pars, trans, tr_pars, g, g_pars, ...
                                x_ref_b, seed_k + b - 1);
            X_paths(:,:,:,b) = outb.sampled_path;
        end

        % mean over all B paths and M chains → 1×T
        mean_cpf = squeeze( mean( mean(X_paths, 4), 3) );

        % per-time squared error vs RTS smoother
        sqerr_rep(k, :) = (mean_cpf - m_s).^2;
    end

    % ---- observed MSE per time (average across the K replicates) ----
    mse_vs_B_T(ib, :) = mean(sqerr_rep, 1);

    % ---- MC "i.i.d." baseline per time: Var(samples)/(M*B) ----
    % pool all M*B samples per time t and estimate their variance
    mc_var_t = zeros(1, T);
    for t = 1:T
        xb = reshape(X_paths(1, t, :, :), 1, M*B);    % NOTE: uses last replicate's X_paths size
        % Recompute xb for all K replicates to be precise:
        % Collect all samples across K:
        all_samples = zeros(1, M*B*K);
        for k = 1:K
            % You can cache per-replicate X_paths if you want; here we approximate
            % with the last replicate's variance as a cheap proxy:
            % (If you want exact, store X_paths per replicate and concat here.)
            all_samples((k-1)*M*B + (1:M*B)) = xb;    %#ok<AGROW>
        end
        mc_var_t(t) = var(all_samples, 1);            % population var
    end
    mc_pred_vs_B_T(ib, :) = mc_var_t / (M * B);
end

%% ===== Plot section (choose t_sel here) =====
% Choose any time index/indices to collapse to a single curve
t_sel =10;                     % e.g., t_sel = 50; or t_sel = [40 80 120];
% Reduce per-time arrays to one scalar per B by averaging over chosen times
mse_vs_B   = mean(mse_vs_B_T(:, t_sel),   2);
mc_pred_vs_B = mean(mc_pred_vs_B_T(:, t_sel), 2);

% Log–log plot
figure;
loglog(Bs, mse_vs_B, 'o-', 'LineWidth',1.5, 'MarkerSize',6); hold on; grid on;
loglog(Bs, mc_pred_vs_B, 's--', 'LineWidth',1.5, 'MarkerSize',6);
xlabel('B (CPF iterations)');
ylabel(sprintf('MSE at t=%s', mat2str(t_sel)));
title(sprintf('CPF mean MSE vs B (K=%d reps, M=%d chains per rep)', K, M));
legend('Observed MSE', 'MC variance/(M·B)', 'Location','southwest');

% Optional reference slopes
C1 = mse_vs_B(end) * sqrt(Bs(end));
loglog(Bs, C1 ./ sqrt(Bs), ':');                     % ~1/sqrt(B)
C2 = mse_vs_B(end) * Bs(end);
loglog(Bs, C2 ./ Bs, ':');                           % ~1/B

% Slope fit
p = polyfit(log(Bs(:)), log(mse_vs_B(:)), 1);
fprintf('Observed MSE slope (log–log) at t=%s: %.3f\n', mat2str(t_sel), p(1));

end
%%
% In the following "if " we have the test for the particle filter, we are
% able to get the MC rate in terms of the number of particles.
if true

%% toy 1D LGSSM
T=10; N=500000; M=300; d_x=1; d_y=1;
rho=0.95; q=0.2^2; r=0.3^2; H=1;

% data
x_true=zeros(1,T); y=zeros(1,T);
rng(0); 
for t=2:T, x_true(t)=rho*x_true(t-1)+sqrt(q)*randn; end
y(1,:)=H*x_true + sqrt(r)*randn(1,T);

% user functions
in_pars.mu = 0;  in_pars.Sigma = 0;  % just for bookkeeping
in_dist = @(p,N,M) reshape( zeros(1,N*M), 1, N, M );

tr_pars.rho=rho; tr_pars.sig=sqrt(q);
trans = @(Xprev,p,t) p.rho*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

g_pars.H=H; g_pars.R=r;
g = @(yt,Xt,p,t) -0.5*((yt - p.H*Xt).^2)/p.R - 0.5*log(2*pi*p.R);


a=5;b=16;
pf_means=zeros(b-a+1,T,M);

pf_m2s   = zeros(K, T, M);


eNes=2.^(a:b);
for i=1:size(eNes,2)

    N=eNes(i);
    out = pf_parallel(y, T, N, M, in_dist, in_pars, trans, tr_pars, g, g_pars, 42);
    w=exp(out.logw);
    pf_means(i,:,:) = squeeze(sum(out.particles.*reshape(w,1,N,T,M),2));   % T × M


    % weighted 2nd moment: E[X_t^2 | y_1:t]
    X2 = out.particles.^2;                        % d_x×N×T×M (here d_x=1)
    pf_m2s(i,:,:) = squeeze( sum(X2 .* reshape(w,1,N,T,M), 2) );               % T×M
    
end
%%
%%  --- Kalman filter (1D) to compare against PF ---
% model:  x_t = rho * x_{t-1} + w_t,     w_t ~ N(0, q)
%         y_t = H   * x_t       + v_t,   v_t ~ N(0, r)
% order:  update at t (using y_t), then predict to t+1
m0 = 0; P0 = 0;
[m_kf, P_kf] = kalman_1d(y, rho, q, r, H, m0, P0);

%% --- PF filtered means (mean over particles), then MC error vs KF ---
% out.particles is d_x × N × T × M; here d_x=1, so we get T×M after squeeze.
w=exp(out.logw);
pf_mean = squeeze(sum(out.particles.*reshape(w,1,N,T,M),2));   % T × M

% Error matrix (T × M): each column is a PF replicate's error path
% Inputs assumed from your loop:
%   pf_means : K × T × M    (K = number of N values)
%   a, b     : define N = 2.^(a:b)
%   m_kf     : T × 1 or 1 × T   (Kalman filtered mean)

Ns = 2.^(a:b);                 % 1 × K
K  = numel(Ns);
[~, T, M] = size(pf_means);

% Ensure m_kf is T×1
m_kf = m_kf(:);                % T × 1

% Error tensor: K × T × M (PF mean minus KF mean)
err = bsxfun(@minus, pf_means, reshape(m_kf.', 1, T, 1));

% MSE over M (replicates), per N and time: K × T
mse_over_M = mean(err.^2, 3);

% Collapse over time to one scalar per N (time-averaged MSE): K × 1
mse = squeeze(mean(mse_over_M, 2));   % K × 1

% --- log–log plot ---
figure; 
loglog(Ns, mse, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
grid on; xlabel('N (particles)'); ylabel('MSE (avg over M, then over time)');
title('PF MSE vs N');

% Optional: add a 1/N reference slope line (anchored at the largest N)
C = mse(end) * Ns(end);                 % choose C so ref matches last point
ref = C ./ Ns;
loglog(Ns, ref, '--');
legend('MSE', 'C / N', 'Location', 'southwest');

% Optional: estimate slope on the log–log plot (should be ~ -1 if ~1/N)
p = polyfit(log(Ns(:)), log(mse(:)), 1);
fprintf('Fitted slope (log–log): %.3f (expect ~ -1 if MSE ~ 1/N)\n', p(1));
%%

% Choose time index (can be a scalar or a vector)
t0 = 3;                        % example: single time; use t0 = [20 50 80] for multiple

% Ensure shapes
K  = numel(Ns);
m_kf = m_kf(:);                 % T×1

% Error at time(s) t0 for each N and each replicate m
% pf_means: K×T×M  →  select T=t0 → K×|t0|×M
err_mean = pf_means(:, t0, :) - reshape(m_kf(t0), 1, numel(t0), 1);

% MSE over M (for each N and each chosen time)
mse_mean_over_M = squeeze( mean( err_mean.^2, 3 ) );      % K×|t0|

% If multiple times are chosen, average across those times to get one curve
mse_mean = mean(mse_mean_over_M, 2);                     % K×1

% --- log–log plot ---
figure; 
loglog(Ns, mse_mean, 'o-', 'LineWidth',1.5, 'MarkerSize',6); hold on; grid on;
xlabel('N (particles)'); ylabel(sprintf('MSE of E[X_t | y_{1:t}] at t=%s', mat2str(t0)));
title('PF Mean MSE vs N (time-selectable)');

% Reference 1/N line
C = mse_mean(end) * Ns(end);
loglog(Ns, C./Ns, '--');
legend('MSE', 'C / N', 'Location','southwest');

% Slope fit
p = polyfit(log(Ns(:)), log(mse_mean(:)), 1);
fprintf('Mean MSE slope (log–log) at t=%s: %.3f\n', mat2str(t0), p(1));
%%
t0=2;
% Kalman second moment
m2_kf = m_kf.^2 + P_kf(:);      % T×1

% PF second-moment error at time(s) t0
err_m2 = pf_m2s(:, t0, :) - reshape(m2_kf(t0), 1, numel(t0), 1);

% MSE over M, then (optionally) average over chosen times
mse_m2_over_M = squeeze( mean( err_m2.^2, 3 ) );          % K×|t0|
mse_m2 = mean(mse_m2_over_M, 2);                          % K×1

% --- log–log plot ---
figure; 
loglog(Ns, mse_m2, 's-', 'LineWidth',1.5, 'MarkerSize',6); hold on; grid on;
xlabel('N (particles)'); ylabel(sprintf('MSE of E[X_t^2 | y_{1:t}] at t=%s', mat2str(t0)));
title('PF Second-Moment MSE vs N (time-selectable)');

% Reference 1/N line
C2 = mse_m2(end) * Ns(end);
loglog(Ns, C2./Ns, '--');
legend('MSE (2nd moment)', 'C / N', 'Location','southwest');

% Slope fit
p2 = polyfit(log(Ns(:)), log(mse_m2(:)), 1);
fprintf('Second-moment MSE slope (log–log) at t=%s: %.3f\n', mat2str(t0), p2(1));

%% (Optional) quick sanity plots
 figure; plot(1:T,m_kf, 'k-', 'LineWidth',1), hold on; plot(1:T,pf_mean(1:end,:), 'Color',[.6 .6 .9]); 
 legend('Kalman','PF means (each m)'); title('Filtered means');
 %figure; histogram(rmse_per_filter, 20); title('RMSE across M filters'); xlabel('RMSE'); ylabel('count');
end
%%
%% -------- local function: 1D Kalman filter --------
