
%% Test for the error of the unbiased score method.
clear; clc; rng(1);
cd ../../../../../../../../../Downloads/cso_first_example_refactored
%% Add local function files to path
thisDir = fileparts(which('cso_first_example_driver.m'));

if isempty(thisDir)
    thisDir = pwd;
end

srcDir = fullfile(thisDir, 'src');
addpath(genpath(srcDir));
rehash;

assert(exist('score_gaussian_ssm', 'file') == 2, ...
    'score_gaussian_ssm.m was not found. Check that src/ is in the same folder as cso_first_example_driver.m.');

%% 1) Model & synthetic data (Gaussian SSM)
rng(2);
T          = 30;
theta_true = 0.7;
q_true     = 0.25^2;
r_true     = 0.2^2;
S0_true    = q_true/(1 - theta_true^2);

% We find from multiple runs that these are 
% the parameters that maximize the likelihood.
%theta_opt=0.888804;
%q_opt=0.028590;
%r_opt0.054407
x = zeros(1,T);
x(1) = sqrt(S0_true)*randn;
for t = 2:T
    x(t) = theta_true*x(t-1) + sqrt(q_true)*randn;
end
y = x + sqrt(r_true)*randn(1,T);

 %% 2) Analytical Gaussian score (for reference later)
[sc_anal, ~] = score_gaussian_ssm(y, theta_true, q_true, r_true, S0_true);
g_anal = [sc_anal.dtheta; sc_anal.dq; sc_anal.dr];
%% 3) PF / PG building blocks
N = 10;   % particles
M = 2;    % parallel chains per PG iteration
% Initial distribution sampler
in_pars.mu    = 0;
in_pars.S0 = S0_true;
in_dist_samp  = @(p,N_,M_) reshape( ...
                        p.mu + sqrt(p.S0)*randn(1,N_*M_), ...
                        1, N_, M_);
trans_pars0.theta = theta_true;
trans_pars0.q     = q_true;
trans_pars0.sig   = sqrt(q_true);
% Transition sampler: x_t = theta x_{t-1} + sqrt(q)*eps
trans_dist_samp = @(Xprev,p,t) ...
    p.theta * Xprev + p.sig * randn(size(Xprev),'like',Xprev);

% Gaussian observation log-likelihood
g_gauss = @(yt,Xt,p,t) reshape( ...
    -0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
    1, size(Xt,2), size(Xt,3));

% t-Student observation log-likelihood (must already exist)
g_t = @(yt,Xt,p,t) log_g_t_stud(yt, Xt, p, t);  % 1×N×M

% Transition log-pdf for backward simulation
trans_logpdf_fun = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - pars.theta*X_prev).^2)/pars.q ...
    - 0.5*log(2*pi*pars.q);    % 1×N

%% 4) Debiasing setup (Rhee–Glynn)
B0   = 2;          % base PG links
Lmax = 6;           % levels {0,1} (you can increase later)
eLes = 0:Lmax;
Bs   = B0 * 2.^(0:Lmax);   % [B0, 2B0, ...] if Lmax>1
l_raw  = (eLes+4).*log(eLes+4).^2 ./ 2.^eLes;
l_dist = l_raw / sum(l_raw);   % pmf on {eLes}

%% 5) Large nu for t-Student
nu_large = 5;
%% 6) Unbiased estimator handles: [dθ; dq; dr] in (theta,q,r)

cpf_choice  = "cpf";        % or "cpf_parallel"
traj_mode   = "backward";   % or "ancestors"
store_parts = false;
store_anc   = false;
store_logw  = false;
N_init      = N;
init_mode   = "weighted";
% Gaussian unbiased estimator
unb_gauss = @(theta,q,r,seed) pg_unbiased_score_gauss( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, struct('theta',theta,'q',q,'sig',sqrt(q)), ...
        g_gauss, struct('R',r), ...
        theta, q, r, S0_true, ...
        Bs, l_dist, seed, ...
        cpf_choice, traj_mode, trans_logpdf_fun, ...
        store_parts, store_anc, store_logw, ...
        N_init, init_mode );

% t-Student unbiased estimator (ν large)
unb_tstud = @(theta,q,r,seed) pg_unbiased_score_tstud( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, struct('theta',theta,'q',q,'sig',sqrt(q)), ...
        g_t, struct('v',nu_large,'sigma',sqrt(r)), ...
        trans_logpdf_fun, ...
        B0, eLes, l_dist, seed, ...
        cpf_choice, traj_mode, ...
        store_parts, store_anc, store_logw, ...
        N_init, init_mode );

unb_tstud2 = @(theta,q,r,nu,seed) pg_unbiased_score_tstud( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, struct('theta',theta,'q',q,'sig',sqrt(q)), ...
        g_t, struct('v',nu,'sigma',sqrt(r)), ...
        trans_logpdf_fun, ...
        B0, eLes, l_dist, seed, ...
        cpf_choice, traj_mode, ...
        store_parts, store_anc, store_logw, ...
        N_init, init_mode );



%% 7) SA settings (in variables (theta, log q, log r))
K_SA  =4*50;                  
theta0 = theta_true-0.2;
q0     =q_true-0.1^2;
r0     = r_true+0.1^2;              % keep r near true
theta0 = 0.5;
r0     =0.2;
q0     = 0.3;              % keep r near true
Gamma = 5*[1; 20; 20]/T;         % step-size scales for [theta; log q; log r]
alpha = 0.5;
n0    = 100;
S_SA  = 5;                    % # of unbiased draws per SA step

seed_SA_mix = 65328290;
C=2;
%%
tic;
m_mix= 3;
for it=1:C
    seed_SA_mix=ceil(abs(100000*sin(10000*it)));
    trace_mix_all(it) = sa_pg_unbiased_mixture_clean( ...
    theta0, q0, r0, ...
    K_SA, ...
    Gamma, alpha, n0, ...
    m_mix, S_SA, ...
    unb_gauss, unb_tstud, ...
    seed_SA_mix);
end
toc
tic;
%%
K_SA  =10;                  
m_star= 3;
for it=1:C
    seed_SA_mix=ceil(abs(100000*sin(10000*it)));
    trace_mix_all(it) = sa_pg_unbiased_model_average_clean( ...
    theta0, q0, r0, ...
    K_SA, ...
    Gamma, alpha, n0, ...
    m_star, S_SA, ...
    unb_gauss, unb_tstud2, ...
    seed_SA_mix);
end
toc

%%
% 7.5) 
% For the biased SA, we want to use the same initial parameters (theta0,q0,r0)
% and as many arguments as possible shared with the setup above.

% Build an observation-parameter struct for the mixture:
% (you can adapt this to the exact interface of sa_pg_mix_ssm)
obs_pars_mix0.R     = r0;                % Gaussian variance
obs_pars_mix0.v     = nu_large;         % t-Student df (large => ~Gaussian)
obs_pars_mix0.sigma = sqrt(r0);         % t-Student scale
m_mix=3;
obs_pars_mix0.m_mix = m_mix;            % mixture parameter
trans_pars0.theta = theta0;
trans_pars0.q     = q0;
trans_pars0.sig   = sqrt(q0);
% If your sa_pg_mix_ssm is defined roughly as:
%   trace = sa_pg_mix_ssm(y,T,N,M,B, ...
%                         in_dist_samp,in_pars, ...
%                         trans_dist_samp,trans_pars0, ...
%                         g_mix,obs_pars_mix0, ...
%                         seed0, ...
%                         cpf_choice,traj_mode,trans_logpdf_fun, ...
%                         store_parts,store_anc,store_logw, ...
%                         N_init,init_mode, ...
%                         K_SA,alpha,Gamma,theta_max, ...
%                         theta0,q0,r0)
% then the following call is consistent:
K=K_SA;
B_pg    = B0*2^(Lmax+6);          % use same base B as in debiasing
seed0_l = seed_SA_mix; % reuse same base seed
g_mix = @(yt,Xt,p,t) g_mix_t_gauss(yt, Xt, p, t);  % mixture log-lik
%trace_mix_lik = sa_pg_mix_ssm( ...
%    y, T, ...
%    N, M, B_pg, ...
%    in_dist_samp, in_pars, ...
%    trans_dist_samp, trans_pars0, ...
%    g_mix, obs_pars_mix0, ...
%    seed0_l, ...
%    cpf_choice, traj_mode, trans_logpdf_fun, ...
%    false,false,false, ...          % store_particles, store_anc, store_logw
%    N_init, init_mode, ...
%    K, alpha, 5*Gamma);                % initial (physical) parameters

%% 7.5.1) Run C biased SA paths (sa_pg_mix_ssm), analogous to section 7
                   % # of unbiased draws per SA step
%K=4*1000;
K=10;
B_pg    = B0*2^(Lmax);
%trace_mix_lik_all = repmat(struct(), 1, C);  % preallocate struct array
%clear("trace_mix_lik_all");
tic;
for it = 1:C

    % Same seed pattern as section 7 (so runs are comparable)
    seed_SA_mix = ceil(abs(100000*sin(1000000*it)));

    % (Optional) keep trans/obs pars consistent each run
    obs_pars_mix0.R     = r0;
    obs_pars_mix0.v     = nu_large;
    obs_pars_mix0.sigma = sqrt(r0);
    obs_pars_mix0.m_mix = m_mix;

    trans_pars0.theta = theta0;
    trans_pars0.q     = q0;
    trans_pars0.sig   = sqrt(q0);

    % Run biased SA
    trace_mix_lik_all(it) = sa_pg_mix_ssm( ...
        y, T, ...
        N, M, B_pg, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, trans_pars0, ...
        g_mix, obs_pars_mix0, ...
        seed_SA_mix, ...
        cpf_choice, traj_mode, trans_logpdf_fun, ...
        false,false,false, ...
        N_init, init_mode, ...
        K, alpha, 5*Gamma);
end
toc

%% 8) theta vs r plots 
figure;
hold on; grid on;
paths=zeros(3,C,K_SA+1);
%paths_mix_lik=zeros(3,K+1);
%paths_mix_lik(1,:)=trace_mix_lik.theta;
%paths_mix_lik(2,:)=trace_mix_lik.q;
%paths_mix_lik(3,:)=trace_mix_lik.r;
mark=0;
for it = 1:C
    theta_path = trace_mix_all(it).theta;  % <-- adjust name if needed
    r_path     = trace_mix_all(it).r;      % <-- adjust name if needed
    q_path     = trace_mix_all(it).q;      % <-- adjust name if needed
    paths(1,it,:)=theta_path;
    paths(2,it,:)=q_path;
    paths(3,it,:)=r_path;
    % If you want log r instead:
    % r_path = log(trace_mix_all(it).r);

    if mark==0
    plot(theta_path, r_path, 'DisplayName', sprintf('path %d', it), ...
        'LineWidth', 1.5, 'MarkerSize', 4);
    else 
        plot(theta_path, q_path, 'DisplayName', sprintf('path %d', it), ...
        'LineWidth', 1.5, 'MarkerSize', 4);
    end
end
    %if mark==0

    %    plot(trace_mix_lik.theta,trace_mix_lik.r , 'b-s', ...
    %    'LineWidth', 2.5, 'MarkerSize', 8);
    %else
    %    plot(trace_mix_lik.theta,trace_mix_lik.q , 'b-s', ...
    %    'LineWidth', 2.5, 'MarkerSize', 8);
    %end

xlabel('\theta');

if mark==0
    ylabel('r');            % or '\log r' if you use log
    title('SA paths in (\theta, r) space');
else
    ylabel('q');            % or '\log r' if you use log
    title('SA paths in (\theta, q) space');
end
legend('show', 'Location', 'best');
hold off;



%%
figure; hold on; grid on;
mark=0;
% Collect paths (3 × C × (K+1)) for: [theta; q; r]
paths_mix_lik_all = zeros(3, C, K+1);
for it = 1:C
    paths_mix_lik_all(1,it,:) = trace_mix_lik_all(it).theta(:).';
    paths_mix_lik_all(2,it,:) = trace_mix_lik_all(it).q(:).';
    paths_mix_lik_all(3,it,:) = trace_mix_lik_all(it).r(:).';
end

% Plot each biased SA path     

for it = 1:C
    theta_path = squeeze(paths_mix_lik_all(1,it,:));
    q_path     = squeeze(paths_mix_lik_all(2,it,:));
    r_path     = squeeze(paths_mix_lik_all(3,it,:));

    if  mark==0
        plot(theta_path, r_path, 'b-o', ...
            'DisplayName', sprintf('biased path %d', it), ...
            'LineWidth', 1.5, 'MarkerSize', 4);
    else
        plot(theta_path, q_path, 'b-o', ...
            'DisplayName', sprintf('biased path %d', it), ...
            'LineWidth', 1.5, 'MarkerSize', 4);
    end
end

xlabel('\theta');
if mark==0
    ylabel('r');
    title('Biased SA paths (7.5.1) in (\theta, r) space');
else
    ylabel('q');
    title('Biased SA paths (7.5.1) in (\theta, q) space');
end

legend('show','Location','best');
hold off;
paths_mix_lik_all;
mean_paths_mix_lik_all=mean(paths_mix_lik_all,2);
mean_paths_mix_lik_all;
sprintf("%7f",mean_paths_mix_lik_all(1:end,end))
%%
%trace_mix_lik_all.theta(end) %0.8077
%trace_mix_lik_all.r(end)     %0.0411
%trace_mix_lik_all.q(end)     %0.0304

theta_end_vals = arrayfun(@(s) s.theta(end), trace_mix_lik_all);
mean_theta_end = mean(theta_end_vals)

r_end_vals = arrayfun(@(s) s.r(end), trace_mix_lik_all);
mean_r_end = mean(r_end_vals)

q_end_vals = arrayfun(@(s) s.q(end), trace_mix_lik_all);
mean_q_end = mean(q_end_vals)




%%
for it = 1:C
    theta_path = trace_mix_all(it).theta;  % <-- adjust name if needed
    r_path     = trace_mix_all(it).r;      % <-- adjust name if needed
    q_path     = trace_mix_all(it).q;      % <-- adjust name if needed
    paths(1,it,:)=theta_path;
    paths(2,it,:)=q_path;
    paths(3,it,:)=r_path;
end


mean(paths(:,:,end-1),2)
%%
f=figure;
%theta_opt=0.888804;
%q_opt=0.028590;
%r_opt=0.054407;

theta_opt=0.8979;
q_opt=   0.0259;
r_opt=0.0564;
pars_opt=[theta_opt,q_opt,r_opt].';
MSE=mean(((pars_opt-paths)./pars_opt).^2,2);
MSE=squeeze(MSE);
loglog(1:(K_SA+1),MSE,"LineWidth",3.5);
grid on;
set(gca, 'FontSize', 25);
legend('\mu','\Sigma^2','\sigma^2', ...
       'Location','southwest');
title(sprintf('Relative MSE of the parameters in terms of SA iterations (\\nu=%.1e)', nu_large));
xlabel('SA iteration K');
ylabel('relative MSE');
%out_file_t = fullfile(fig_path, ...
%    sprintf('relMSE_total_nu_%1.1e_lw_25.pdf', nu_large));
%exportgraphics(f, out_file_t, 'ContentType','vector');

%% Assume: MSE is of size [n_comp x n_paths x n_iter]
[n_comp, n_paths, n_iter] = size(MSE);

its = 1:n_iter;   % SA iterations (adjust if needed)

% Average over independent paths (dim 2) -> [n_comp x n_iter]
MSE_mean = squeeze(mean(MSE, 2));

lw  = 2.5;    % line width
fsL = 25;     % label font size
fsT = 25;     % title font size
fsA = 25;     % axes tick font size
fig_path = ['/Users/alvarem/Library/CloudStorage/GoogleDrive-', ...
    'miguelangel.alvarezballesteros@kaust.edu.sa/Other computers/', ...
    'My MacBook Pro/MEGA/0KAUST/0CSO_project/Figs'];
%% 1) Component 1 (theta)
f=figure;
loglog(its, MSE_mean(1, :), 'LineWidth', lw);
grid on;
xlabel('SA iteration', 'FontSize', fsL, 'Interpreter', 'latex');
ylabel('$\varepsilon_{\theta}^2$', 'FontSize', fsL, 'Interpreter', 'latex');
%title('MSE of $\theta$', 'FontSize', fsT, 'Interpreter', 'latex');
set(gca, 'FontSize', fsA);
out_file = fullfile(fig_path, ...
        sprintf('SA_MSE_component_theta.pdf'));
%    exportgraphics(f, out_file, 'ContentType','vector');
%% 2) Component 2 (q)
f=figure;
loglog(its, MSE_mean(2, :), 'LineWidth', lw);
grid on;
xlabel('SA iteration', 'FontSize', fsL, 'Interpreter', 'latex');
ylabel('$\varepsilon_{q}^2$', 'FontSize', fsL, 'Interpreter', 'latex');
%title('MSE of $q$', 'FontSize', fsT, 'Interpreter', 'latex');
set(gca, 'FontSize', fsA);
out_file = fullfile(fig_path, ...
        sprintf('SA_MSE_component_q.pdf'));
%    exportgraphics(f, out_file, 'ContentType','vector');
%% 3) Component 3 (r)
f=figure;
loglog(its, MSE_mean(3, :), 'LineWidth', lw);
grid on;
xlabel('SA iteration', 'FontSize', fsL, 'Interpreter', 'latex');
ylabel('$\varepsilon_{r}^2$', 'FontSize', fsL, 'Interpreter', 'latex');
%title('MSE of $r$', 'FontSize', fsT, 'Interpreter', 'latex');
set(gca, 'FontSize', fsA);
out_file = fullfile(fig_path, ...
        sprintf('SA_MSE_component_r.pdf'));
 %   exportgraphics(f, out_file, 'ContentType','vector');
%% 4) Sum over components
mse_sum = sum(MSE_mean, 1);   % [1 x n_iter]

f=figure;
loglog(its, mse_sum, 'LineWidth', lw);
grid on;
xlabel('SA iteration', 'FontSize', fsL, 'Interpreter', 'latex');
ylabel('$\varepsilon_{{sum}}^2$', 'FontSize', fsL, 'Interpreter', 'latex');
%title('Sum of component-wise MSE', 'FontSize', fsT, 'Interpreter', 'latex');
set(gca, 'FontSize', fsA);
out_file = fullfile(fig_path, ...
        sprintf('SA_MSE_component_sum.pdf'));
%    exportgraphics(f, out_file, 'ContentType','vector');
%% Test of the unbiased score vs likelihood mixture methods.clear; clc; rng(1);

%% 1) Model & synthetic data (Gaussian SSM)
rng(2);
T          = 10;
theta_true = 0.7;
q_true     = 0.2^2;
r_true     = 0.3^2;
S0_true    = q_true/(1 - theta_true^2);

x = zeros(1,T);
x(1) = sqrt(S0_true)*randn;
for t = 2:T
    x(t) = theta_true*x(t-1) + sqrt(q_true)*randn;
end
y = x + sqrt(r_true)*randn(1,T);

%% 2) Analytical Gaussian score (for reference later)
[sc_anal, ~] = score_gaussian_ssm(y, theta_true, q_true, r_true, S0_true);
g_anal = [sc_anal.dtheta; sc_anal.dq; sc_anal.dr];

%% 3) PF / PG building blocks

N = 10;   % particles
M = 2;    % parallel chains per PG iteration

% Initial distribution sampler
in_pars.mu    = 0;
in_pars.S0 = S0_true;
in_dist_samp  = @(p,N_,M_) reshape( ...
                        p.mu + sqrt(p.S0)*randn(1,N_*M_), ...
                        1, N_, M_);
trans_pars0.theta = theta_true;
trans_pars0.q     = q_true;
trans_pars0.sig   = sqrt(q_true);
% Transition sampler: x_t = theta x_{t-1} + sqrt(q)*eps
trans_dist_samp = @(Xprev,p,t) ...
    p.theta * Xprev + p.sig * randn(size(Xprev),'like',Xprev);

% Gaussian observation log-likelihood
g_gauss = @(yt,Xt,p,t) reshape( ...
    -0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
    1, size(Xt,2), size(Xt,3));

% t-Student observation log-likelihood (must already exist)
g_t = @(yt,Xt,p,t) log_g_t_stud(yt, Xt, p, t);  % 1×N×M

% Transition log-pdf for backward simulation
trans_logpdf_fun = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - pars.theta*X_prev).^2)/pars.q ...
    - 0.5*log(2*pi*pars.q);    % 1×N

%% 4) Debiasing setup (Rhee–Glynn)
B0   = 30;          % base PG links
Lmax = 2;           % levels {0,1} (you can increase later)
eLes = 0:Lmax;
Bs   = B0 * 2.^(0:Lmax);   % [B0, 2B0, ...] if Lmax>1

l_raw  = (eLes+4).*log(eLes+4).^2 ./ 2.^eLes;
l_dist = l_raw / sum(l_raw);   % pmf on {eLes}

%% 5) Large nu for t-Student
nu_large = 5;
%% 6) Unbiased estimator handles: [dθ; dq; dr] in (theta,q,r)
cpf_choice  = "cpf";        % or "cpf_parallel"
traj_mode   = "backward";   % or "ancestors"
store_parts = false;
store_anc   = false;
store_logw  = false;
N_init      = N;
init_mode   = "weighted";

% Gaussian unbiased estimator
unb_gauss = @(theta,q,r,seed) pg_unbiased_score_gauss( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, struct('theta',theta,'q',q,'sig',sqrt(q)), ...
        g_gauss, struct('R',r), ...
        theta, q, r, S0_true, ...
        Bs, l_dist, seed, ...
        cpf_choice, traj_mode, trans_logpdf_fun, ...
        store_parts, store_anc, store_logw, ...
        N_init, init_mode );

% t-Student unbiased estimator (ν large)
unb_tstud = @(theta,q,r,seed) pg_unbiased_score_tstud( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, struct('theta',theta,'q',q,'sig',sqrt(q)), ...
        g_t, struct('v',nu_large,'sigma',sqrt(r)), ...
        trans_logpdf_fun, ...
        B0, eLes, l_dist, seed, ...
        cpf_choice, traj_mode, ...
        store_parts, store_anc, store_logw, ...
        N_init, init_mode );

unb_tstud2 = @(theta,q,r,nu,seed) pg_unbiased_score_tstud( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, struct('theta',theta,'q',q,'sig',sqrt(q)), ...
        g_t, struct('v',nu,'sigma',sqrt(r)), ...
        trans_logpdf_fun, ...
        B0, eLes, l_dist, seed, ...
        cpf_choice, traj_mode, ...
        store_parts, store_anc, store_logw, ...
        N_init, init_mode );


%% 7) SA settings (in variables (theta, log q, log r))
K_SA  = 40;                  
theta0 = theta_true;
q0     =q_true;
r0     = r_true;              % keep r near true
Gamma = 5*[1; 0; 5]/T;         % step-size scales for [theta; log q; log r]
alpha = 0.5;
n0    = 100;
S_SA  = 100;                    % # of unbiased draws per SA step

seed_SA_mix = 8347;





m_star     = 1;
trace_mix = sa_pg_unbiased_model_average_clean( ...
    theta0, q0, r0, ...
    K_SA, ...
    Gamma, alpha, n0, ...
    m_star, S_SA, ...
    unb_gauss, unb_tstud, ...
    seed_SA_mix);



%% 7.5) Biased SA with mixture likelihood: sa_pg_mix_ssm

% For the biased SA, we want to use the same initial parameters (theta0,q0,r0)
% and as many arguments as possible shared with the setup above.

% Build an observation-parameter struct for the mixture:
% (you can adapt this to the exact interface of sa_pg_mix_ssm)
obs_pars_mix0.R     = r0;                % Gaussian variance
obs_pars_mix0.v     = nu_large;         % t-Student df (large => ~Gaussian)
obs_pars_mix0.sigma = sqrt(r0);         % t-Student scale
obs_pars_mix0.m_star = m_star;            % mixture parameter

% If your sa_pg_mix_ssm is defined roughly as:
%   trace = sa_pg_mix_ssm(y,T,N,M,B, ...
%                         in_dist_samp,in_pars, ...
%                         trans_dist_samp,trans_pars0, ...
%                         g_mix,obs_pars_mix0, ...
%                         seed0, ...
%                         cpf_choice,traj_mode,trans_logpdf_fun, ...
%                         store_parts,store_anc,store_logw, ...
%                         N_init,init_mode, ...
%                         K_SA,alpha,Gamma,theta_max, ...
%                         theta0,q0,r0)
% then the following call is consistent:

B_pg    = B0*2^Lmax;          % use same base B as in debiasing
seed0_l = seed_SA_mix; % reuse same base seed
g_mix = @(yt,Xt,p,t) g_mix_t_gauss(yt, Xt, p, t);  % mixture log-lik
trace_mix_lik = sa_pg_mix_ssm( ...
    y, T, ...
    N, M, B_pg, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars0, ...
    g_mix, obs_pars_mix0, ...
    seed0_l, ...
    cpf_choice, traj_mode, trans_logpdf_fun, ...
    false,false,false, ...          % store_particles, store_anc, store_logw
    N_init, init_mode, ...
    K_SA, alpha, Gamma);                % initial (physical) parameters






% NOTE:
% If your actual sa_pg_mix_ssm signature is slightly different,
% adapt the ordering of obs_pars_mix0, seed, Gamma, theta0,q0,r0 accordingly.
%% 8) Analytical Gaussian gradient field in (theta, log q) with r fixed

in = 1;  % index from which you trust the SA path
theta_all = [];
lq_all    = [];

% Unbiased SA path (mixture)
theta_all = [theta_all; trace_mix.theta(in:end)];
lq_all    = [lq_all;    log(trace_mix.q(in:end))];

% Biased SA path (likelihood-based mixture), if available
if exist('trace_mix_lik','var') && isfield(trace_mix_lik,'theta') && isfield(trace_mix_lik,'q')
    theta_all = [theta_all; trace_mix_lik.theta(in:end)];
    lq_all    = [lq_all;    log(trace_mix_lik.q(in:end))];
end

% Add true parameter to ensure it's inside the frame
theta_all = [theta_all; theta_true];
lq_all    = [lq_all;    log(q_true)];

margin_theta = 0.02;
margin_lq    = 0.2;

th_lo = min(theta_all) - margin_theta;
th_hi = max(theta_all) + margin_theta;

lq_lo = min(lq_all)    - margin_lq;
lq_hi = max(lq_all)    + margin_lq;

% if you want roughly square axes:
if (th_hi - th_lo) < (lq_hi - lq_lo)
    % expand theta range a bit
    mid_th = 0.5*(th_lo+th_hi);
    half   = 0.5*(lq_hi-lq_lo);
    th_lo  = mid_th - half;
    th_hi  = mid_th + half;
end

n_grid = 400;
theta_grid = linspace(th_lo, th_hi, n_grid);
lq_grid    = linspace(lq_lo, lq_hi, n_grid);
[TH, LQ]   = meshgrid(theta_grid, lq_grid);

GRAD_TH = zeros(size(TH));
GRAD_LQ = zeros(size(TH));
ELL     = zeros(size(TH));

for i = 1:n_grid
    for j = 1:n_grid
        th = TH(i,j);
        qg = exp(LQ(i,j));  % q = e^{log q}

        [sc,~,ell] = score_gaussian_ssm_new(y, th, qg, r_true, S0_true);
        ELL(i,j)      = ell;
        GRAD_TH(i,j)  = sc.dtheta;      % ∂ℓ/∂θ
        GRAD_LQ(i,j)  = qg * sc.dq;     % ∂ℓ/∂(log q)
    end
end
grad_norm = hypot(GRAD_TH, GRAD_LQ);


%% 9) Plot: analytical score field + SA paths in (theta, log q)

figure;
axis equal;
pbaspect([1 1 1]);
contour(TH, LQ, ELL, 60, 'LineWidth', 1); hold on; grid on;
quiver(TH, LQ, GRAD_TH, GRAD_LQ, 'k');

% Unbiased SA path (Rhee–Glynn mixture score)
plot(trace_mix.theta, log(trace_mix.q), 'r-o', ...
    'LineWidth', 1.3, 'MarkerSize', 4, ...
    'DisplayName','SA path (unbiased mixture)');

% Biased SA path (likelihood-based mixture)
if exist('trace_mix_lik','var') && isfield(trace_mix_lik,'theta') && isfield(trace_mix_lik,'q')
    plot(trace_mix_lik.theta, log(trace_mix_lik.q), 'b-s', ...
        'LineWidth', 1.3, 'MarkerSize', 4, ...
        'DisplayName','SA path (biased mixture)');
end

% True parameter
plot(theta_true, log(q_true), 'ks', ...
    'MarkerFaceColor','k','MarkerSize',7, ...
    'DisplayName','true (\theta,q)');

xlabel('\theta','Interpreter','latex');
ylabel('\log q','Interpreter','latex');
title({'Analytical Gaussian score field in $(\theta,\log q)$ with $r$ fixed', ...
       'SA paths (mixture: unbiased vs biased)'}, ...
      'Interpreter','latex');

legend('Location','best','Interpreter','latex');

%% 10) Log-likelihood contours and gradient field in (theta, r) with q fixed

% Assumptions:
% - y, q_true, r_true, S0_true are in workspace
% - score_gaussian_ssm_new(y, theta, q, r, S0) returns [score, parts, loglik]
% - Optional: trace_gauss.r, trace_tstud.r exist from SA runs

% 1) Collect all theta and r values to define a plotting window
theta_all_r = [];
r_all       = [];

if exist('trace_gauss','var') && isfield(trace_mix,'theta') && isfield(trace_mix,'r')
    theta_all_r = [theta_all_r; trace_mix.theta(:)];
    r_all       = [r_all;       trace_mix.r(:)];
end


theta_all_r = [theta_all_r; theta_true];
r_all       = [r_all;       r_true];

% Small margins
margin_theta_r = 0.02;
margin_r       = 0.1 * r_true;   % 10% of true r as margin

th_lo_r = 0.45;
th_hi_r = 0.9;

r_lo    = 0.04;  % keep r>0
r_hi    = 0.14;

% 2) Build grid in (theta, r)
n_grid_r   = 200;
theta_grid_r = linspace(th_lo_r, th_hi_r, n_grid_r);
r_grid       = linspace(r_lo,   r_hi,    n_grid_r);
[TH_r, R]    = meshgrid(theta_grid_r, r_grid);

% 3) Evaluate log-likelihood and gradient on the grid
ELL_r    = zeros(size(TH_r));   % log-likelihood values
GRAD_THr = zeros(size(TH_r));   % dℓ/dθ
GRAD_R   = zeros(size(TH_r));   % dℓ/dr

for i = 1:n_grid_r
    for j = 1:n_grid_r
        th = TH_r(i,j);
        rr = R(i,j);
        [sc, ~, ell] = score_gaussian_ssm_new(y, th, q_true, rr, S0_true);
        ELL_r(i,j)    = ell;
        GRAD_THr(i,j) = sc.dtheta;
        GRAD_R(i,j)   = sc.dr;
    end
end

% 4) Plot: log-likelihood contours + gradient field
%figure;
%contour(TH_r, R, ELL_r, 30, 'LineWidth', 1); hold on; grid on;
%colormap turbo;
%colorbar;
%title({'Log-likelihood contours in $(\theta, r)$ with $q$ fixed'}, ...
%      'Interpreter','latex');
%
% Gradient field
%quiver(TH_r, R, GRAD_THr, GRAD_R, 'k');
%
% Mark the true parameter
%plot(theta_true, r_true, 'ks', ...
%    'MarkerFaceColor','k','MarkerSize',7, ...
%    'DisplayName','true $(\theta,r)$');
%
%xlabel('\theta','Interpreter','latex');
%ylabel('r','Interpreter','latex');
%legend('Location','best','Interpreter','latex');
%axis tight;


%% 11) Add SA paths in (theta, r) on top of log-likelihood field

figure;
contour(TH_r, R, ELL_r, 200, 'LineWidth', 1); hold on; grid on;
colormap turbo;
colorbar;
title({'Log-likelihood contours in $(\theta, r)$ with $q$ fixed'}, ...
      'Interpreter','latex');

% Gradient field
quiver(TH_r, R, GRAD_THr, GRAD_R, 'k', 'AutoScale','on');

% Unbiased SA path (mixture)
if exist('trace_mix','var') && isfield(trace_mix,'theta') && isfield(trace_mix,'r')
    plot(trace_mix.theta, trace_mix.r, 'r-o', ...
        'LineWidth', 1.5, 'MarkerSize', 4, ...
        'DisplayName','SA path (unbiased mixture)');
end

% Biased SA path (likelihood-based mixture)
if exist('trace_mix_lik','var') && isfield(trace_mix_lik,'theta') && isfield(trace_mix_lik,'r')
    plot(trace_mix_lik.theta, trace_mix_lik.r, 'b-s', ...
        'LineWidth', 1.5, 'MarkerSize', 4, ...
        'DisplayName','SA path (biased mixture)');
end

% True parameter
plot(theta_true, r_true, 'ks', ...
    'MarkerFaceColor','k','MarkerSize',7, ...
    'DisplayName','true $(\theta,r)$');

xlabel('\theta','Interpreter','latex');
ylabel('r','Interpreter','latex');
legend('Location','best','Interpreter','latex');
axis tight;

sprintf("The family trace is: %.4f",trace_mix.family)
%%
mean(trace_mix.family)


%% ===== SA experiment for mixture SSM (PG + mixture score) =====
clear; clc;

%% --- 1. True model parameters ---
T      = 50;
theta_true = 0.95;
q_true     = 0.2^2;
r_g_true   = 0.3^2;
r_t_true   = 0.3^2;
nu_true    = 5;
m_mix_true = 2;
S0_true    = q_true/(1-theta_true^2);

%% --- 2. Generate latent and mixture observations ---
rng(10);
x = zeros(1,T); x(1) = sqrt(S0_true)*randn;
for t=2:T
    x(t) = theta_true*x(t-1) + sqrt(q_true)*randn;
end

y = zeros(1,T);
for t=1:T
    % mixture: choose component
    if rand < 1/(m_mix_true+1)
        % t-Student component
        eps_t = trnd(nu_true,1,1);   % if you don't have stats toolbox, replace by your own
        y(t) = x(t) + sqrt(r_t_true)*eps_t;
    else
        % Gaussian component
        y(t) = x(t) + sqrt(r_g_true)*randn;
    end
end

%% --- 3. PF/PG functions ---
in_pars.mu    = 0;
in_pars.S0 = S0_true;
in_dist_samp  = @(p,N,M) reshape(p.mu + sqrt(p.Sigma)*randn(1,N*M),1,N,M);

trans_dist_samp = @(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

%% --- 4. SA settings ---
N     = 10;
M     = 2;
B     = 10;        % links per SA iteration
K     = 100;       % SA steps
seed0 = 123;

% initial parameters (perturbed)
theta0 = 0.8;
q0     = 0.15^2;
r_t0   = 0.4^2;
r_g0   = 0.4^2;

gamma_vec = [0.2; 5; 2; 2];   % base step sizes [θ; log q; log r_t; log r_g]
alpha     = 0.6;

cpf_choice = "cpf";           % or "cpf_parallel" if you want
traj_mode  = "backward";      % as before

%% --- 5. Run SA ---
trace_mix = sa_pg_mix_ssm( ...
    y, T, ...
    N, M, B, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, ...
    theta0, q0, r_t0, r_g0, nu_true, m_mix_true, ...
    S0_true, ...
    gamma_vec, alpha, ...
    K, seed0, ...
    cpf_choice, traj_mode);

%% --- 6. Plot parameter traces vs iteration ---
iters = 0:K;

figure;
subplot(2,2,1);
plot(iters, trace_mix.theta, 'o-'); hold on; yline(theta_true,'r--');
grid on;
xlabel('n'); ylabel('\theta_n');
title('SA trace: \theta');

subplot(2,2,2);
plot(iters, trace_mix.q, 'o-'); hold on; yline(q_true,'r--');
grid on;
xlabel('n'); ylabel('q_n');
title('SA trace: q');

subplot(2,2,3);
plot(iters, trace_mix.r_t, 'o-'); hold on; yline(r_t_true,'r--');
grid on;
xlabel('n'); ylabel('r_{t,n}');
title('SA trace: r_t (t-comp var)');

subplot(2,2,4);
plot(iters, trace_mix.r_g, 'o-'); hold on; yline(r_g_true,'r--');
grid on;
xlabel('n'); ylabel('r_{g,n}');
title('SA trace: r_g (Gaussian var)');
sgtitle('SA with PG + mixture score (non-unbiased)');


%% ===== TEST: score_mix_from_paths_vectorized =====
clear; clc;

%% --- 1. Model parameters (1D AR(1) SSM) ---
T_true      = 30;
theta_true  = 0.95;
q_true      = 0.2^2;
r_g_true    = 0.3^2;       % Gaussian obs variance
r_t_true    = 0.3^2;       % t-Student "variance" via sigma_t
nu_true     = 10;          % df for t
m_mix_true  = 2;           % mixture parameter m (so w_t=1/(m+1), w_g=m/(m+1))

S0_true     = q_true/(1-theta_true^2);   % stationary initial var

%% --- 2. Generate latent state and observations under some regime ---
rng(123);
x = zeros(1, T_true);
x(1) = sqrt(S0_true)*randn;
for t = 2:T_true
    x(t) = theta_true * x(t-1) + sqrt(q_true)*randn;
end

% For this simple test, let's generate purely Gaussian observations
% (just to have something): y = x + N(0, r_g_true)
y = x + sqrt(r_g_true)*randn(1, T_true);

%% --- 3. Build "paths" X_paths (1×T×P) ---
% In a real run, these would come from PG. For test, we build P noisy versions
P = 20;
X_paths = zeros(1, T_true, P);
for p = 1:P
    X_paths(1,:,p) = x + 0.2*randn(1, T_true);   % perturb true path
end

%% --- 4. Define parameter structs for score_mix_from_paths_vectorized ---
% Initial Gaussian (N(0, S0))
init_pars_test.mu    = 0;
init_pars_test.S0 = S0_true;

% Transition Gaussian
trans_pars_test.theta = theta_true;
trans_pars_test.q     = q_true;
trans_pars_test.sig   = sqrt(q_true);

% Mixture observation parameters
obs_pars_mix_test.v       = nu_true;
obs_pars_mix_test.sigma_t = sqrt(r_t_true);  % scale for t
obs_pars_mix_test.R_g     = r_g_true;        % Gaussian variance
obs_pars_mix_test.m_mix   = m_mix_true;

%% --- 5. Call score_mix_from_paths_vectorized ---

S_mix = score_mix_from_paths_vectorized( ...
    y, X_paths, init_pars_test, trans_pars_test, obs_pars_mix_test);

%% --- 6. Inspect results ---
fprintf('=== score_mix_from_paths_vectorized test ===\n');
fprintf('Size S_mix.per_path_init   : %s\n', mat2str(size(S_mix.per_path_init)));
fprintf('Size S_mix.per_path_trans  : %s\n', mat2str(size(S_mix.per_path_trans)));
fprintf('Size S_mix.per_path_obs_t  : %s\n', mat2str(size(S_mix.per_path_obs_t)));
fprintf('Size S_mix.per_path_obs_g  : %s\n', mat2str(size(S_mix.per_path_obs_g)));

fprintf('\nAvg init     = %.4e\n', S_mix.avg_init);
fprintf('Avg trans(1) = dtheta = %.4e\n', S_mix.avg_trans(1));
fprintf('Avg trans(2) = dq     = %.4e\n', S_mix.avg_trans(2));
fprintf('Avg obs_t    = d/d log sigma_t  = %.4e\n', S_mix.avg_obs_t);
fprintf('Avg obs_g    = d/d log sigma_g  = %.4e\n', S_mix.avg_obs_g);

%% ===== Mixture-at-model-level SSM + PGibbs visualization =====
clear; clc;

%% 1) Model parameters

T          = 30;
theta_true = 0.95;
q_true     = 0.2^2;
r_true     = 0.3^2;
S0_true    = q_true/(1-theta_true^2);

% mixture parameter m: P(t-Student model) = 1/(m+1), P(Gaussian model) = m/(m+1)
m_mix   = 1;                  % e.g. P(t) = 1/4, P(Gauss) = 3/4
nu_t    = 1;                 % df of t-Student
sigma_t = sqrt(r_true);       % choose scale so variance is comparable to r

%% 2) Simulate hidden state
rng(43);
x_true = zeros(1,T);
x_true(1) = sqrt(S0_true)*randn;
for t = 2:T
    x_true(t) = theta_true*x_true(t-1) + sqrt(q_true)*randn;
end

%% 3) Choose observation family ONCE and generate y_{1:T}
rng(6);
y = zeros(1,T);

u = rand;
if u < 1/(m_mix + 1)
    obs_family = "t-student";   % all times t use t-Student
else
    obs_family = "gaussian";    % all times t use Gaussian
end

switch obs_family
    case "t-student"
        % sample entire observation sequence from t-Student around x_t
        for t = 1:T
            % sample t_nu via N / sqrt(Chi2/nu) (no toolbox)
            z  = randn;
            u_chi2 = sum(randn(nu_t,1).^2);
            t_samp = z / sqrt(u_chi2 / nu_t);
            y(t) = x_true(t) + sigma_t * t_samp;
        end

    case "gaussian"
        % sample entire observation sequence from N(x_t, r_true)
        y = x_true + sqrt(r_true)*randn(1,T);
end

%% 4) PF/CPF/PG building blocks
% ---- initial distribution: X_1 ~ N(0, S0_true) ----
in_pars.mu    = 0;
in_pars.Sigma = S0_true;
in_dist_samp  = @(p,N,M) reshape(p.mu + sqrt(p.Sigma)*randn(1,N*M), 1, N, M);

% ---- state transition: X_t = theta * X_{t-1} + sqrt(q)*eps ----
tr_pars.theta = theta_true;
tr_pars.q     = q_true;
tr_pars.sig   = sqrt(q_true);

trans_dist_samp = @(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

% ---- mixture observation log-likelihood (t-Student + Gaussian) ----
g_pars_mix.R     = r_true;
g_pars_mix.v     = nu_t;
g_pars_mix.sigma = sigma_t;
g_pars_mix.m_mix = m_mix;

g_mix = @(yt,Xt,p,t) g_mix_t_gauss(yt, Xt, p, t);   % mixture log-lik

% ---- transition log pdf for backward simulation ----
trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - pars.theta*X_prev).^2)/pars.q ...
    - 0.5*log(2*pi*pars.q);   % 1×N

%% 5) Run one PGibbs with mixture observations

N      = 10;          % particles
M      = 3;           % chains in parallel
B      = 40;          % links per chain
seed0  = 123;
cpf_choice  = "cpf";         % or "cpf_parallel"
traj_mode   = "backward";
store_parts = false; store_anc = false; store_logw = false;
N_init      = N;     init_mode = "weighted";

out_pg = pgibbs_run_init_pfmean( ...
    y, T, N, M, B, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, tr_pars, ...
    g_mix, g_pars_mix, ...
    seed0, ...
    cpf_choice, traj_mode, trans_logpdf, ...
    store_parts, store_anc, store_logw, ...
    N_init, init_mode);

% out_pg.X_paths: 1×T×M×(B+1).
% Take the last link of the first chain for plotting
X_paths_all = out_pg.X_paths;
[~, ~, M_, Bp1] = size(X_paths_all); %#ok<ASGLU>
m_sel = 1;
b_sel = Bp1;
x_pg  = squeeze(X_paths_all(1,:,m_sel,b_sel));  % 1×T

%% 6) Plot: true state, PG path, and observations

t_axis = 1:T;

figure; hold on; grid on;
plot(t_axis, x_true, '-k', 'LineWidth', 2.0);        % true latent state
plot(t_axis, x_pg,   '-r', 'LineWidth', 2.0);        % PG sampled path

% observations: single style, but color encode family
switch obs_family
    case "t-student"
        plot(t_axis, y, 'bo', 'MarkerSize', 7, 'LineWidth', 1.5);
    case "gaussian"
        plot(t_axis, y, 'gs', 'MarkerSize', 7, 'LineWidth', 1.5);
end

xlabel('time t');
ylabel('value');
title(sprintf(['Mixture *model-level* observations: all %s, m=%d, \\\\nu=%d'], ...
    obs_family, m_mix, nu_t), 'Interpreter','tex');

if obs_family == "t-student"
    leg_obs = 'Obs (all t-Student)';
else
    leg_obs = 'Obs (all Gaussian)';
end

legend('True state x_t', ...
       'PG sampled path', ...
       leg_obs, ...
       'Location','best');
hold off;


%% Mixture parameters pgibbs_run_init_pfmean
% Observation parameters
r_true   = 0.3^2;
nu_mix   = 10;              % or whatever df you want
sigma_t  = sqrt(r_true);    % often you match the Gaussian variance
m=2;
m_mix    = m;               % your mixing parameter
B=10;
g_pars_mix.R     = r_true;
g_pars_mix.v     = nu_mix;
g_pars_mix.sigma = sigma_t;
g_pars_mix.m_mix = m_mix;
%%
N = 10;   % particles
M = 2;    % parallel chains per PG iteration

% Initial distribution sampler
in_pars.mu    = 0;
in_pars.S0 = S0_true;
in_dist_samp  = @(p,N_,M_) reshape( ...
                        p.mu + sqrt(p.S0)*randn(1,N_*M_), ...
                        1, N_, M_);

% Transition sampler: x_t = theta x_{t-1} + sqrt(q)*eps
trans_dist_samp = @(Xprev,p,t) ...
    p.theta * Xprev + p.sig * randn(size(Xprev),'like',Xprev);

% Gaussian observation log-likelihood
g_gauss = @(yt,Xt,p,t) reshape( ...
    -0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
    1, size(Xt,2), size(Xt,3));

% t-Student observation log-likelihood (must already exist)
g_t = @(yt,Xt,p,t) log_g_t_stud(yt, Xt, p, t);  % 1×N×M

% Transition log-pdf for backward simulation
trans_logpdf_fun = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - pars.theta*X_prev).^2)/pars.q ...
    - 0.5*log(2*pi*pars.q);    % 1×N
store_parts = false;
store_anc   = false;
store_logw  = false;
%%

% Example PG run with mixture observation model
seed0=1254;
out_pg = pgibbs_run_init_pfmean( ...
    y, T, N, M, B, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    @g_mix_t_gauss, g_pars_mix, ...    % <-- NEW g and pars
    seed0, ...
    cpf_choice, traj_mode, trans_logpdf_fun, ...
    store_particles, store_ancestors, store_logw, ...
    N_init, init_mode);
%%


% Test on the limit of the t-student when nu->infinity
v=1e10;
% As v->infinity the two following expressions must be the same
gammaln((v+1)/2) - gammaln(v/2)- 0.5*log(v*pi)
-log(2*pi)/2
%%
z=1:10;
sigma=2;
v=1e10;
% As v->infinity the two following expressions must be the same
- 0.5*(v+1) .* log(1 + z ./ (v*sigma^2))   % 1×N×M
-z ./ (2*sigma^2)

%%
%% ==============================================================
% SA for Gaussian SSM using PG-based UNBIASED scores
% - Compare Gaussian vs t-Student (large nu) SA paths
% - Overlay on analytical Gaussian gradient field in (theta, log q)
% ==============================================================

clear; clc; rng(1);

%% 1) Model & synthetic data (Gaussian SSM)
rng(2);
T          = 30;
theta_true = 0.7;
q_true     = 0.2^2;
r_true     = 0.3^2;
S0_true    = q_true/(1 - theta_true^2);

x = zeros(1,T);
x(1) = sqrt(S0_true)*randn;
for t = 2:T
    x(t) = theta_true*x(t-1) + sqrt(q_true)*randn;
end
y = x + sqrt(r_true)*randn(1,T);

%% 2) Analytical Gaussian score (for reference later)
[sc_anal, ~] = score_gaussian_ssm(y, theta_true, q_true, r_true, S0_true);
g_anal = [sc_anal.dtheta; sc_anal.dq; sc_anal.dr];

%% 3) PF / PG building blocks

N = 10;   % particles
M = 2;    % parallel chains per PG iteration

% Initial distribution sampler
in_pars.mu    = 0;
in_pars.S0 = S0_true;
in_dist_samp  = @(p,N_,M_) reshape( ...
                        p.mu + sqrt(p.S0)*randn(1,N_*M_), ...
                        1, N_, M_);

% Transition sampler: x_t = theta x_{t-1} + sqrt(q)*eps
trans_dist_samp = @(Xprev,p,t) ...
    p.theta * Xprev + p.sig * randn(size(Xprev),'like',Xprev);

% Gaussian observation log-likelihood
g_gauss = @(yt,Xt,p,t) reshape( ...
    -0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
    1, size(Xt,2), size(Xt,3));

% t-Student observation log-likelihood (must already exist)
g_t = @(yt,Xt,p,t) log_g_t_stud(yt, Xt, p, t);  % 1×N×M

% Transition log-pdf for backward simulation
trans_logpdf_fun = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - pars.theta*X_prev).^2)/pars.q ...
    - 0.5*log(2*pi*pars.q);    % 1×N

%% 4) Debiasing setup (Rhee–Glynn)
B0   = 1;          % base PG links
Lmax = 3;           % levels {0,1} (you can increase later)
eLes = 0:Lmax;
Bs   = B0 * 2.^(0:Lmax);   % [B0, 2B0, ...] if Lmax>1

l_raw  = (eLes+4).*log(eLes+4).^2 ./ 2.^eLes;
l_dist = l_raw / sum(l_raw);   % pmf on {eLes}

%% 5) Large nu for t-Student
nu_large = 2;

%% 6) Unbiased estimator handles: [dθ; dq; dr] in (theta,q,r)

cpf_choice  = "cpf";        % or "cpf_parallel"
traj_mode   = "backward";   % or "ancestors"
store_parts = false;
store_anc   = false;
store_logw  = false;
N_init      = N;
init_mode   = "weighted";

% Gaussian unbiased estimator
unb_gauss = @(theta,q,r,seed) pg_unbiased_score_gauss( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, struct('theta',theta,'q',q,'sig',sqrt(q)), ...
        g_gauss, struct('R',r), ...
        theta, q, r, S0_true, ...
        Bs, l_dist, seed, ...
        cpf_choice, traj_mode, trans_logpdf_fun, ...
        store_parts, store_anc, store_logw, ...
        N_init, init_mode );

% t-Student unbiased estimator (ν large)
unb_tstud = @(theta,q,r,seed) pg_unbiased_score_tstud( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, struct('theta',theta,'q',q,'sig',sqrt(q)), ...
        g_t, struct('v',nu_large,'sigma',sqrt(r)), ...
        trans_logpdf_fun, ...
        B0, eLes, l_dist, seed, ...
        cpf_choice, traj_mode, ...
        store_parts, store_anc, store_logw, ...
        N_init, init_mode );

%% 7) SA settings (in variables (theta, log q, log r))
K_SA  = 100;                  
theta0 = 0.9;
q0     =q_true+0.1;
r0     = (0.2)^2+0.1;              % keep r near true
Gamma = 5*[1; 10; 5]/T;         % step-size scales for [theta; log q; log r]
alpha = 0.2;
n0    = 100;
S_SA  = 2;                    % # of unbiased draws per SA step

seed_SA_gauss = 8347;
seed_SA_t     = 8347;

% (i) Pure Gaussian unbiased SA: m_mix large
m_mix_gauss = 5;
trace_gauss = sa_pg_unbiased_mixture_clean( ...
    theta0, q0, r0, ...
    K_SA, ...
    Gamma, alpha, n0, ...
    m_mix_gauss, S_SA, ...
    unb_gauss, unb_tstud, ...
    seed_SA_gauss);

% (ii) Pure t-Student unbiased SA: m_mix = 0
m_mix_t     = 1;
trace_tstud = sa_pg_unbiased_mixture_clean( ...
    theta0, q0, r0, ...
    K_SA, ...
    Gamma, alpha, n0, ...
    m_mix_t, S_SA, ...
    unb_gauss, unb_tstud, ...
    seed_SA_t);

%% 8) Analytical Gaussian gradient field in (theta, log q) with r fixed
in=1;
theta_all = [trace_gauss.theta(in:end); trace_tstud.theta(in:end)];
lq_all    = [log(trace_gauss.q(in:end)); log(trace_tstud.q(in:end))];
margin_theta = 0.02;
margin_lq    = 0.2;

lq_lo = min(lq_all)    - margin_lq;
lq_hi = max(lq_all)    + margin_lq;
th_lo = min(theta_all) - margin_theta;
th_hi = th_lo+lq_hi-lq_lo;


n_grid = 400;
theta_grid = linspace(th_lo, th_hi, n_grid);
lq_grid    = linspace(lq_lo, lq_hi, n_grid);
[TH, LQ]   = meshgrid(theta_grid, lq_grid);

GRAD_TH = zeros(size(TH));
GRAD_LQ = zeros(size(TH));
ELL = zeros(size(TH));
for i = 1:n_grid
    for j = 1:n_grid
        th = TH(i,j);
        qg = exp(LQ(i,j));  % q = e^{log q}

        [sc,~,ell] = score_gaussian_ssm_new(y, th, qg, r_true, S0_true);
        ELL(i,j) = ell;
        GRAD_TH(i,j) = sc.dtheta;      % ∂ℓ/∂θ
        GRAD_LQ(i,j) = qg * sc.dq;     % ∂ℓ/∂(log q)
    end
end
grad_norm = hypot(GRAD_TH, GRAD_LQ);
%% 9) Plot: analytical score field + 100-step SA paths in (theta, log q)
figure;
axis equal;
pbaspect([1 1 1]);
%contour(TH, LQ, grad_norm, 30, 'LineWidth', 1); hold on; grid on;
contour(TH, LQ, ELL, 60, 'LineWidth', 1); hold on; grid on;
quiver(TH, LQ, GRAD_TH, GRAD_LQ, 'k');

% SA paths (only first 100+1 points by construction)
plot(trace_gauss.theta, log(trace_gauss.q), 'r-o', ...
    'LineWidth', 1.3, 'MarkerSize', 4, ...
    'DisplayName','SA path (Gaussian unb.)');

plot(trace_tstud.theta, log(trace_tstud.q), 'b-o', ...
    'LineWidth', 1.3, 'MarkerSize', 4, ...
    'DisplayName','SA path (t-Student unb.)');

% True parameter
plot(theta_true, log(q_true), 'ks', ...
    'MarkerFaceColor','k','MarkerSize',7, ...
    'DisplayName','true (\theta,q)');

xlabel('\theta','Interpreter','latex');
ylabel('\log q','Interpreter','latex');
title({'Analytical Gaussian score field in $(\theta,\log q)$ with $r$ fixed', ...
       'SA paths (100 steps) evolving in log-coordinates'}, ...
      'Interpreter','latex');

legend('Location','best','Interpreter','latex');

%% 10) Log-likelihood contours and gradient field in (theta, r) with q fixed

% Assumptions:
% - y, q_true, r_true, S0_true are in workspace
% - score_gaussian_ssm_new(y, theta, q, r, S0) returns [score, parts, loglik]
% - Optional: trace_gauss.r, trace_tstud.r exist from SA runs

% 1) Collect all theta and r values to define a plotting window
theta_all_r = [];
r_all       = [];

if exist('trace_gauss','var') && isfield(trace_gauss,'theta') && isfield(trace_gauss,'r')
    theta_all_r = [theta_all_r; trace_gauss.theta(:)];
    r_all       = [r_all;       trace_gauss.r(:)];
end
if exist('trace_tstud','var') && isfield(trace_tstud,'theta') && isfield(trace_tstud,'r')
    theta_all_r = [theta_all_r; trace_tstud.theta(:)];
    r_all       = [r_all;       trace_tstud.r(:)];
end

theta_all_r = [theta_all_r; theta_true];
r_all       = [r_all;       r_true];

% Small margins
margin_theta_r = 0.02;
margin_r       = 0.1 * r_true;   % 10% of true r as margin

th_lo_r = min(theta_all_r) - margin_theta_r;
th_hi_r = max(theta_all_r) + margin_theta_r;

r_lo    = max( min(r_all) - margin_r,  1e-6 );  % keep r>0
r_hi    = max(r_all) + margin_r;

% 2) Build grid in (theta, r)
n_grid_r   = 200;
theta_grid_r = linspace(th_lo_r, th_hi_r, n_grid_r);
r_grid       = linspace(r_lo,   r_hi,    n_grid_r);
[TH_r, R]    = meshgrid(theta_grid_r, r_grid);

% 3) Evaluate log-likelihood and gradient on the grid
ELL_r    = zeros(size(TH_r));   % log-likelihood values
GRAD_THr = zeros(size(TH_r));   % dℓ/dθ
GRAD_R   = zeros(size(TH_r));   % dℓ/dr

for i = 1:n_grid_r
    for j = 1:n_grid_r
        th = TH_r(i,j);
        rr = R(i,j);
        [sc, ~, ell] = score_gaussian_ssm_new(y, th, q_true, rr, S0_true);
        ELL_r(i,j)    = ell;
        GRAD_THr(i,j) = sc.dtheta;
        GRAD_R(i,j)   = sc.dr;
    end
end

% 4) Plot: log-likelihood contours + gradient field
%figure;
%contour(TH_r, R, ELL_r, 30, 'LineWidth', 1); hold on; grid on;
%colormap turbo;
%colorbar;
%title({'Log-likelihood contours in $(\theta, r)$ with $q$ fixed'}, ...
%      'Interpreter','latex');
%
% Gradient field
%quiver(TH_r, R, GRAD_THr, GRAD_R, 'k');
%
%% Mark the true parameter
%plot(theta_true, r_true, 'ks', ...
%    'MarkerFaceColor','k','MarkerSize',7, ...
%    'DisplayName','true $(\theta,r)$');
%
%xlabel('\theta','Interpreter','latex');
%ylabel('r','Interpreter','latex');
%legend('Location','best','Interpreter','latex');
%axis tight;


%% 11) Add SA paths in (theta, r) on top of log-likelihood field

figure;
contour(TH_r, R, ELL_r, 200, 'LineWidth', 1); hold on; grid on;
colormap turbo;
colorbar;
title({'Log-likelihood contours in $(\theta, r)$ with $q$ fixed'}, ...
      'Interpreter','latex');

% Gradient field (optional)
quiver(TH_r, R, GRAD_THr, GRAD_R, 'k', 'AutoScale','on');

% SA paths (if available)
if exist('trace_gauss','var') && isfield(trace_gauss,'theta') && isfield(trace_gauss,'r')
    plot(trace_gauss.theta, trace_gauss.r, 'r-o', ...
        'LineWidth', 1.5, 'MarkerSize', 4, ...
        'DisplayName','SA path (Gaussian unb.)');
end

if exist('trace_tstud','var') && isfield(trace_tstud,'theta') && isfield(trace_tstud,'r')
    plot(trace_tstud.theta, trace_tstud.r, 'b-o', ...
        'LineWidth', 1.5, 'MarkerSize', 4, ...
        'DisplayName','SA path (t-Student unb.)');
end

% True parameter
plot(theta_true, r_true, 'ks', ...
    'MarkerFaceColor','k','MarkerSize',7, ...
    'DisplayName','true $(\theta,r)$');

xlabel('\theta','Interpreter','latex');
ylabel('r','Interpreter','latex');
legend('Location','best','Interpreter','latex');
axis tight;

%%
%==============================================================
%==============================================================
%==============================================================
%% ==============================================================
%  SA driver: mixture of Gaussian / t-Student unbiased scores
%  (works in variables (theta, log q, log r))
% ==============================================================

%% ===============================================================
%  1) Model & synthetic data (1D Gaussian SSM)
% ================================================================
clear; clc;

T           = 30;
theta_true  = 0.95;
q_true      = 0.2^2;
r_true      = 0.3^2;
S0_true     = q_true/(1 - theta_true^2);    % stationary var

rng(42);
x = zeros(1,T); 
x(1) = sqrt(S0_true)*randn;
for t = 2:T
    x(t) = theta_true*x(t-1) + sqrt(q_true)*randn;
end
y = x + sqrt(r_true)*randn(1,T);

%% ===============================================================
%  2) User functions: PF / CPF building blocks
% ================================================================

% Initial sampler: X1 ~ N(0, S0_true)
in_pars.mu    = 0;
in_pars.Sigma = S0_true;
in_dist_samp  = @(p,N,M) reshape(p.mu + sqrt(p.Sigma)*randn(1,N*M), 1, N, M);

% Transition sampler: X_t = theta * X_{t-1} + sqrt(q)*eps
trans_pars.theta = theta_true;   % these will be overwritten inside pg_unbiased_score_gauss
trans_pars.q     = q_true;
trans_pars.sig   = sqrt(q_true);
trans_dist_samp  = @(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

% Observation log-likelihood: Y_t | X_t ~ N(X_t, r)
g_pars.R = r_true;   % will also be overwritten inside pg_unbiased_score_gauss
g_gauss  = @(yt,Xt,p,t) reshape( ...
                  -0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
                  1, size(Xt,2), size(Xt,3));

% Transition log-density (for backward simulation)
trans_logpdf_fun = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - pars.theta*X_prev).^2)/pars.q - 0.5*log(2*pi*pars.q);  % 1×N

%% ===============================================================
%  3) Analytical Gaussian score at true parameters
% ================================================================
[sc_true, ~, loglik_true] = score_gaussian_ssm_new(y, theta_true, q_true, r_true, S0_true);
g_true = [sc_true.dtheta; sc_true.dq; sc_true.dr];   % 3×1

fprintf('Analytical score at true params:\n');
fprintf('  dtheta = %+ .4e\n', sc_true.dtheta);
fprintf('  dq     = %+ .4e\n', sc_true.dq);
fprintf('  dr     = %+ .4e\n', sc_true.dr);
fprintf('  loglik = %+ .4e\n\n', loglik_true);

%% ===============================================================
%  4) Debiasing design: Bs and l_dist  (Rhee–Glynn levels)
% ================================================================
% Example: geometric B-grid
B0   = 1;                         % base PG length
eLes = 0:10;                        % levels {0,1,2}  → Bs = B0*2^ell
Bs   = B0 * 2.^(0:max(eLes));      % length Lmax+1

% Level distribution l_dist over eLes (must have same length as Bs)
% Example: simple geometric-like weights, normalized:
tmp    = (eLes+4).*log(eLes+4).^2 ./ 2.^eLes;
l_dist = tmp / sum(tmp);

%% ===============================================================
%  5) PG / CPF settings
% ================================================================
N          = 10;          % # particles
M          = 2;           % # chains per PG iteration
cpf_choice = "cpf";       % or "cpf_parallel"
traj_mode  = "backward";  % or "ancestors"
store_particles  = false;
store_ancestors  = false;
store_logw       = false;
N_init     = N;           % particles in initial PF in pgibbs_run_init_pfmean
init_mode  = "weighted";  % use weighted PF mean as starting path
base_seed  = 12345;

%% ===============================================================
%  6) Grid of S values and # repetitions
% ================================================================
% S = number of i.i.d. unbiased estimators we average
S_vals = 2.^(3:7);     % e.g. S in {1,2,4,8,16,32}
K_reps = 100;          % # independent Monte Carlo repetitions

nS    = numel(S_vals);
err   = zeros(3, K_reps, nS);   % errors for (dtheta,dq,dr)

%% ===============================================================
%  7) Monte Carlo loop: for each S, average S unbiased estimators
% ================================================================
fprintf('Running MC experiment: %d S-values, %d reps each...\n', nS, K_reps);
tic;
for sIdx = 1:nS
    S = S_vals(sIdx);
    fprintf('  S = %d\n', S);

    for rep = 1:K_reps

        % Collect S independent unbiased score estimates
        Gs = zeros(3, S);  % columns = 3×1 vectors [dθ; dq; dr]

        for s = 1:S
            seed_s = base_seed + 1000*s + 100000*rep;

            [g_unb, ~] = pg_unbiased_score_gauss( ...
                y, T, N, M, ...
                in_dist_samp, in_pars, ...
                trans_dist_samp, trans_pars, ...
                g_gauss, g_pars, ...
                theta_true, q_true, r_true, S0_true, ...  % physical params
                Bs, l_dist, seed_s, ...                  % debiasing
                cpf_choice, traj_mode, trans_logpdf_fun, ...
                store_particles, store_ancestors, store_logw, ...
                N_init, init_mode);

            % g_unb is 3×1
            Gs(:, s) = g_unb;
        end

        % Average over S
        g_hat = mean(Gs, 2);     % 3×1
        err(:, rep, sIdx) = g_hat - g_true;
    end
end
tot_time = toc;
fprintf('Total time: %.2f s\n\n', tot_time);

%% ===============================================================
%  8) Compute MSE, bias^2, variance per component and total
% ================================================================
mse_comp   = zeros(3, nS);
bias2_comp = zeros(3, nS);
var_comp   = zeros(3, nS);

mse_total   = zeros(1, nS);
bias2_total = zeros(1, nS);
var_total   = zeros(1, nS);

for sIdx = 1:nS
    E = err(:,:,sIdx);   % 3 × K_reps

    % Component-wise
    mse_comp(:,sIdx)   = mean(E.^2, 2);
    bias_comp          = mean(E, 2);
    bias2_comp(:,sIdx) = bias_comp.^2;
    var_comp(:,sIdx)   = mse_comp(:,sIdx) - bias2_comp(:,sIdx);

    % Total (sum over components)
    e_tot = sum(E.^2, 1);            % 1×K_reps
    mse_total(sIdx)   = mean(e_tot); % scalar

    b_tot              = mean(E, 2);           % 3×1
    bias2_total(sIdx)  = sum(b_tot.^2);        % scalar
    var_total(sIdx)    = mse_total(sIdx) - bias2_total(sIdx);
end
%% 9) Relative MSE, bias^2, variance vs S (Gaussian unbiased score)
% g_true = [dtheta_true; dq_true; dr_true];  (3×1)
eps_rel   = 1e-12;
den_comp  = max(g_true.^2, eps_rel);        % 3×1
den_total = max(norm(g_true)^2, eps_rel);   % scalar

% mse_comp, bias2_comp, var_comp are 3×nS
% We want rel_*_comp as nS×3 so we can use rel_*_comp(:,j)
rel_MSE_comp   = (mse_comp.')./den_comp.';    % nS×3
rel_bias2_comp = (bias2_comp.')./den_comp.';  % nS×3
rel_var_comp   = (var_comp.')./den_comp.';    % nS×3

% Total (1×nS)
rel_MSE_total   = mse_total   / den_total;    % 1×nS
rel_bias2_total = bias2_total / den_total;    % 1×nS
rel_var_total   = var_total   / den_total;    % 1×nS

% --- 9.1) Combined figure: components + total ---
comp_names = {'d\theta','dq','dr'};

figure;
for j = 1:3
    subplot(2,2,j);
    loglog(S_vals, rel_MSE_comp(:,j),   'o-','LineWidth',2.5); hold on; grid on;
    loglog(S_vals, rel_bias2_comp(:,j), 's--','LineWidth',2.0);
    loglog(S_vals, rel_var_comp(:,j),   'd--','LineWidth',2.0);

    % 1/S reference (calibrate at largest S) for relative MSE
    Cj_rel = rel_MSE_comp(end,j) * S_vals(end);
    loglog(S_vals, Cj_rel ./ S_vals, ':','LineWidth',2.2);

    xlabel('S (number of unbiased estimators averaged)');
    ylabel(['rel. MSE(',comp_names{j},')']);
    title(['Component: ',comp_names{j}]);
    legend('rel MSE','rel bias^2','rel var','C/S','Location','southwest');
    set(gca, 'FontSize', 14);
end

subplot(2,2,4);
loglog(S_vals, rel_MSE_total,   'o-','LineWidth',2.5); hold on; grid on;
loglog(S_vals, rel_bias2_total, 's--','LineWidth',2.0);
loglog(S_vals, rel_var_total,   'd--','LineWidth',2.0);

Ctot_rel = rel_MSE_total(end) * S_vals(end);
loglog(S_vals, Ctot_rel ./ S_vals, ':','LineWidth',1.2);

xlabel('S (number of unbiased estimators averaged)');
ylabel('Total relative MSE');
title('Total score relative MSE (Gaussian unbiased PG)');
legend('rel MSE','rel bias^2','rel var','C/S','Location','southwest');
set(gca, 'FontSize', 14);
sgtitle('Relative MSE vs S (Gaussian unbiased score)');

%% 9.5) Separate plots of relative MSE components and total (export as PDF)

fig_path = ['/Users/alvarem/Library/CloudStorage/GoogleDrive-', ...
    'miguelangel.alvarezballesteros@kaust.edu.sa/Other computers/', ...
    'My MacBook Pro/MEGA/0KAUST/0CSO_project/Figs'];



% ensure directory exists
if ~exist(fig_path, 'dir')
    mkdir(fig_path);
end

% --- Component names and file tags ---
comp_names = {'d\mu','d(\Sigma^2)','d(\sigma^2)'};
file_tag   = {'relMSEvsS_gauss_dmu', ...
              'relMSEvsS_gauss_d1Sigma2', ...
              'relMSEvsS_gauss_d2sigma2'};

%% --- 9.5.1 Component-wise figures ---
for j = 1:3
    f = figure;
    loglog(S_vals, rel_MSE_comp(:,j),   'o-','LineWidth',3.5); hold on; grid on;
    loglog(S_vals, rel_bias2_comp(:,j), 's--','LineWidth',3.0);
    loglog(S_vals, rel_var_comp(:,j),   'd--','LineWidth',3.0);

    % 1/S reference (scaled at largest S)
    Cj_rel = rel_MSE_comp(end,j) * S_vals(end);
    loglog(S_vals, Cj_rel ./ S_vals, ':','LineWidth',3.2);

    xlabel('S (number of unbiased estimators averaged)');
    ylabel(['rel. MSE(', comp_names{j}, ')']);
    title(['Component: ', comp_names{j}, ' (Gaussian)']);
    legend('rel MSE','rel bias^2','rel var','C/S', ...
        'Location','southwest', 'FontSize', 25);

    set(gca, 'FontSize', 25);

    % Export
    out_file = fullfile(fig_path, ...
        sprintf('relMSE_component_%s_gauss.pdf', file_tag{j}));
 %   exportgraphics(f, out_file, 'ContentType','vector');
end

%% --- 9.5.2 Total relative MSE figure ---
f_tot = figure;
loglog(S_vals, rel_MSE_total,   'o-','LineWidth',3.5); hold on; grid on;
loglog(S_vals, rel_bias2_total, 's--','LineWidth',3.0);
loglog(S_vals, rel_var_total,   'd--','LineWidth',3.0);

Ctot_rel = rel_MSE_total(end) * S_vals(end);
loglog(S_vals, Ctot_rel ./ S_vals, ':','LineWidth',3.2);

xlabel('S (number of unbiased estimators averaged)');
ylabel('Total relative MSE');
title('Total score relative MSE (Gaussian)');

legend('rel MSE','rel bias^2','rel var','C/S', ...
       'Location','southwest','Interpreter','latex', 'FontSize', 25);

set(gca, 'FontSize', 25);

% Export
out_file_tot = fullfile(fig_path, 'relMSE_total_gauss.pdf');
%exportgraphics(f_tot, out_file_tot, 'ContentType','vector');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ===== Test: MSE of averaged t-Student and gaussian unbiased estimators vs S and vs cost =====
clear; clc; rng(42);
%% 1) Model & synthetic data (Gaussian SSM, same as before)
T          = 30;
theta_true = 0.95;
q_true     = 0.2^2;
r_true     = 0.3^2;
S0_true    = q_true/(1 - theta_true^2);
% Latent AR(1)
x = zeros(1,T); 
x(1) = sqrt(S0_true)*randn;
for t = 2:T
    x(t) = theta_true*x(t-1) + sqrt(q_true)*randn;
end

% Gaussian observations (true model)
y = x + sqrt(r_true)*randn(1,T);

%% 2) Analytical Gaussian score (ground truth for comparison)

[sc_gauss, ~] = score_gaussian_ssm(y, theta_true, q_true, r_true, S0_true);
g_true = [sc_gauss.dtheta; sc_gauss.dq; sc_gauss.dr];  % 3×1

fprintf('Analytical Gaussian score:\n');
fprintf('  dtheta = %+ .4e\n', sc_gauss.dtheta);
fprintf('  dq     = %+ .4e\n', sc_gauss.dq);
fprintf('  dr     = %+ .4e\n', sc_gauss.dr);

%% 3) PF / PG ingredients with t-Student observations (large nu)

% Initial distribution for PF/CPF
in_pars.mu    = 0;
in_pars.S0 = S0_true;
in_dist_samp  = @(p,N,M) reshape( ...
                        p.mu + sqrt(p.S0)*randn(1,N*M), ...
                        1, N, M);

% Transition
trans_pars.theta = theta_true;
trans_pars.q     = q_true;
trans_pars.sig   = sqrt(q_true);
trans_dist_samp  = @(Xprev,p,t) ...
    p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

% t-Student observation: large nu, scale = sqrt(r_true)
nu_large         = 1e5;
obs_pars_t.v     = nu_large;
obs_pars_t.sigma = sqrt(r_true);

% t-Student log-likelihood for PF/CPF
g_t = @(yt,Xt,p,t) log_g_t_stud(yt,Xt,p,t);   % must return 1×N×M

% Transition log-density for backward CPF/PG
trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - theta_true*X_prev).^2)/q_true ...
    - 0.5*log(2*pi*q_true);   % 1×N

%% 4) Unbiased estimator configuration (Rhee-Glynn levels)

B0    = 1;              % base number of PG links for level 0
eLes  = 0:10;             % example levels {0,1}; extend if desired
l_raw = (eLes+4).*log(eLes+4).^2 ./ 2.^eLes;
l_dist = l_raw / sum(l_raw);   % normalize to sum to 1

% PG / PF configuration inside each unbiased estimator
N      = 10;             % particles for PG
M      = 2;              % parallel chains per PG call
cpf_choice = "cpf";      % or "cpf_parallel"
traj_mode  = "backward"; % recommended with trans_logpdf
store_parts = false;
store_anc   = false;
store_logw  = false;
N_init      = N;
init_mode   = "weighted";

%% 5) Outer MSE test over S (number of i.i.d. unbiased estimators averaged)

% S-values (number of unbiased estimators per average)
S_vals = 2.^(3:7);       % e.g. S in {1,2,4,8,...,64}
KS     = numel(S_vals);

% Number of independent "averaging experiments" per S (to estimate MSE)
Krep   = 100;
% Storage
err_comp  = zeros(Krep, KS, 3);   % component-wise errors [dtheta,dq,dr]
err_total = zeros(Krep, KS);      % ||error||^2

% Theoretical expected cost per unbiased estimator:
% cost(L) = Bs(end) = B0 * 2^L, and L ~ (eLes, l_dist)
expected_cost_per_est = sum( B0 * 2.^eLes .* l_dist );
avg_cost_S = S_vals * expected_cost_per_est;   % expected cost for averaging S estimators

seed_base = 12345;

fprintf('\nRunning MSE test for t-Student unbiased estimator (nu = %.1e)...\n', nu_large);
tic;
for iS = 1:KS
    S = S_vals(iS);

    for k = 1:Krep

        % Collect S independent unbiased estimators of the 3×1 score
        ests = zeros(3, S);
        for s = 1:S
            seed0 = seed_base + 100000*iS + 1000*k + s;  % decorrelate seeds
            ests(:,s) = pg_unbiased_score_tstud( ...
                y, T, N, M, ...
                in_dist_samp, in_pars, ...
                trans_dist_samp, trans_pars, ...
                g_t, obs_pars_t, ...
                trans_logpdf, ...
                B0, eLes, l_dist, ...
                seed0, ...
                cpf_choice, traj_mode, ...
                store_parts, store_anc, store_logw, ...
                N_init, init_mode);
        end

        % Average over S estimators
        g_hat = mean(ests, 2);           % 3×1

        % Error vs analytical Gaussian score
        err_vec = g_hat - g_true;        % 3×1
        err_comp(k, iS, :) = err_vec;    % store component-wise
        err_total(k, iS)   = sum(err_vec.^2);  % squared norm
    end

    fprintf('  S = %4d done.\n', S);
end
toc;
%% 5.5) 

g_hat_store = err_comp + reshape(g_true, 1, 1, 3);   % Krep x KS x 3
% g_hat_store: Krep x KS x 3
size(g_hat_store)
W_total = 0;
sum_weighted = zeros(1,1,3);

for iS = 1:KS
    S = S_vals(iS);
    % sum over replicates for this S
    sum_iS = sum(g_hat_store(:,iS,:), 1);     % 1 x 1 x 3
    sum_weighted = sum_weighted + S * sum_iS; % weight by S
    W_total = W_total + S * Krep;
end

g_pool = squeeze(sum_weighted) / W_total;     % 3x1 pooled mean

% err_comp: Krep x KS x 3
% g_true : 3x1
% g_pool : 3x1

delta = reshape(g_true - g_pool, 1, 1, 3);     % 1x1x3 for broadcasting

err_comp_pool = err_comp + delta;              % Krep x KS x 3

% Update err_total consistently (squared norm per replicate)
err_total_pool = squeeze(sum(err_comp_pool.^2, 3));   % Krep x KS


%% 6) Compute MSE, bias^2, variance vs S
% Component-wise MSE, bias^2, var for each S and each component
MSE_comp   = squeeze( mean(err_comp_pool.^2, 1) );                    % KS × 3
bias_comp  = squeeze( mean(err_comp_pool, 1) );                       % KS × 3
bias2_comp = bias_comp.^2;                                       % KS × 3
var_comp   = MSE_comp - bias2_comp;                              % KS × 3

% Total MSE, bias^2, var (sum over components)
MSE_total   = squeeze( mean(err_total_pool, 1) );                     % KS × 1
mean_err    = squeeze( mean(err_comp_pool, 1) ).';                    % 3×KS
bias2_total = sum(mean_err.^2, 1).';                             % KS × 1
var_total = sum(var_comp, 2);   % KS×1, much more stable
%% 6.5 (Pre-step) Relative MSE, bias^2 and variance
g_pool2      = (g_pool(:)).^2;        % 3x1
norm_g_pool2 = sum(g_pool2);          % scalar

% KS x 3 (implicit expansion works in recent MATLAB; otherwise use bsxfun)
rel_MSE_comp   = MSE_comp   ./ (g_pool2.'); 
rel_bias2_comp = bias2_comp ./ (g_pool2.');
rel_var_comp   = var_comp   ./ (g_pool2.');

% KS x 1
rel_MSE_total   = MSE_total   / norm_g_pool2;
rel_MSE_total=rel_MSE_total';
rel_bias2_total = bias2_total / norm_g_pool2;
rel_var_total = rel_MSE_total - rel_bias2_total;

%% 7) Plot: MSE vs S with 1/S reference (total and components)
comp_names = {'d\theta','dq','dr'};
figure;
for j = 1:3
    subplot(2,2,j);
    loglog(S_vals, MSE_comp(:,j), 'o-','LineWidth',1.5); hold on; grid on;
    loglog(S_vals, bias2_comp(:,j), 's--','LineWidth',1.0);
    loglog(S_vals, var_comp(:,j),  'd--','LineWidth',1.0);

    % 1/S reference (calibrate at largest S)
    Cj = MSE_comp(end,j) * S_vals(end);
    loglog(S_vals, Cj ./ S_vals, ':','LineWidth',1.2);

    xlabel('S (number of unbiased estimators averaged)');
    ylabel(['MSE(',comp_names{j},')']);
    title(['Component: ',comp_names{j}]);
    legend('MSE','bias^2','var','C/S','Location','southwest');
end
subplot(2,2,4);
loglog(S_vals, MSE_total, 'o-','LineWidth',1.5); hold on; grid on;
loglog(S_vals, bias2_total, 's--','LineWidth',1.0);
loglog(S_vals, var_total,  'd--','LineWidth',1.0);
Ctot = MSE_total(end) * S_vals(end);
loglog(S_vals, Ctot ./ S_vals, ':','LineWidth',1.2);
xlabel('S'); ylabel('Total MSE');
title('Total score MSE (sum of components)');
legend('MSE','bias^2','var','C/S','Location','southwest');
sgtitle(sprintf('t-Student unbiased score (nu=%.1e) vs Gaussian analytical score', nu_large));

%% 8) Plot: MSE vs expected cost (total)
figure;
loglog(avg_cost_S, MSE_total, 'o-','LineWidth',1.6); hold on; grid on;
Ccost = MSE_total(end) * avg_cost_S(end);
loglog(avg_cost_S, Ccost ./ avg_cost_S, '--','LineWidth',1.2);
xlabel('Expected cost = S * E[cost per estimator]');
ylabel('Total MSE');
title(sprintf('MSE vs expected cost (t-Student unbiased, \\nu=%.1e)', nu_large));
legend('MSE','C / cost','Location','southwest');
%% 9) Plot: relative MSE vs S with 1/S reference (total and components)
comp_names = {'d\mu','d(\Sigma^2)','d(\sigma^2)'};
figure;
for j = 1:3
    subplot(2,2,j);
    loglog(S_vals, rel_MSE_comp(:,j), 'o-','LineWidth',1.5); hold on; grid on;
    loglog(S_vals, rel_bias2_comp(:,j), 's--','LineWidth',1.0);
    loglog(S_vals, rel_var_comp(:,j),  'd--','LineWidth',1.0);

    % 1/S reference (calibrate at largest S) for relative MSE
    Cj_rel = rel_MSE_comp(end,j) * S_vals(end);
    loglog(S_vals, Cj_rel ./ S_vals, ':','LineWidth',1.2);

    xlabel('S (number of unbiased estimators averaged)');
    ylabel(['rel. MSE(',comp_names{j},')']);
    title(['Component: ',comp_names{j}]);
    legend('rel MSE','rel bias^2','rel var','C/S','Location','southwest');
end

subplot(2,2,4);
loglog(S_vals, rel_MSE_total, 'o-','LineWidth',1.5); hold on; grid on;
loglog(S_vals, rel_bias2_total, 's--','LineWidth',1.0);
loglog(S_vals, rel_var_total,  'd--','LineWidth',1.0);

Ctot_rel = rel_MSE_total(end) * S_vals(end);
loglog(S_vals, Ctot_rel ./ S_vals, ':','LineWidth',1.2);

xlabel('S');
ylabel('Total relative MSE');
title('Total score relative MSE (sum of components)');
legend('rel MSE','rel bias^2','rel var','C/S','Location','southwest');

sgtitle(sprintf('Relative MSE vs S (t-Student unbiased score, \\nu=%.1e)', nu_large));
%% 9.5) Separate plots of relative MSE components and total (export as PDF)

fig_path = ['/Users/alvarem/Library/CloudStorage/GoogleDrive-miguelangel.alvarezballesteros@kaust.edu.sa/', ...
            'Other computers/My MacBook Pro/MEGA/0KAUST/0CSO_project/Figs'];

% ensure directory exists
if ~exist(fig_path, 'dir')
    mkdir(fig_path);
end

comp_names = {'d\mu','d(\Sigma^2)','d(\sigma^2)'};
file_tag   = {'relMSEvsS_tstud_large_v_dmu','relMSEvsS_tstud_large_v_d1Sigma2'...
    ,'relMSEvsS_tstud_large_v_d2sigma2'};   % for filenames

% --- 9.5.1 Component-wise figures ---
for j = 1:3
    f = figure;
    loglog(S_vals, rel_MSE_comp(:,j), 'o-','LineWidth',3.5); hold on; grid on;
    loglog(S_vals, rel_bias2_comp(:,j), 's--','LineWidth',3.0);
    loglog(S_vals, rel_var_comp(:,j),  'd--','LineWidth',3.0);

    % 1/S reference (calibrate at largest S) for relative MSE
    Cj_rel = rel_MSE_comp(end,j) * S_vals(end);
    loglog(S_vals, Cj_rel ./ S_vals, ':','LineWidth',3.2);

    xlabel('S (number of unbiased estimators averaged)');
    ylabel(['rel. MSE(',comp_names{j},')']);
    %ylabel(['rel. MSE(',comp_names{j},')'], 'Interpreter','latex');
    title(['Component: ',comp_names{j}, ...
           sprintf(' (\\nu=%.1f)', nu_large)]);
    legend('rel MSE','rel bias^2','rel var','C/S', ...
           'Location','southwest');

    set(gca, 'FontSize', 25);

    %export to PDF
    out_file = fullfile(fig_path, ...
        sprintf('relMSE_component_%s_nu_%1.1e.pdf', file_tag{j}, nu_large));
 %   exportgraphics(f, out_file, 'ContentType','vector');
end

% --- 9.5.2 Total relative MSE figure ---
f_tot = figure;
loglog(S_vals, rel_MSE_total, 'o-','LineWidth',3.5); hold on; grid on;
loglog(S_vals, rel_bias2_total, 's--','LineWidth',3.0);
loglog(S_vals, rel_var_total,  'd--','LineWidth',3.0);

Ctot_rel = rel_MSE_total(end) * S_vals(end);
loglog(S_vals, Ctot_rel ./ S_vals, ':','LineWidth',3.2);

xlabel('S (number of unbiased estimators averaged)', 'Interpreter','latex');
ylabel('Total relative MSE', 'Interpreter','latex');
title(sprintf('Total score relative MSE (\\nu=%.1f)', nu_large));
legend('rel MSE','rel bias^2','rel var','C/S', ...
       'Location','southwest','Interpreter','latex');

set(gca, 'FontSize', 25);

%export total figure
out_file_tot = fullfile(fig_path, ...
    sprintf('relMSE_total_nu_%1.1e.pdf', nu_large));
%exportgraphics(f_tot, out_file_tot, 'ContentType','vector');
%% 10) Plot: relative MSE vs expected cost (total)
figure;
loglog(avg_cost_S, rel_MSE_total, 'o-','LineWidth',1.6); hold on; grid on;

Ccost_rel = rel_MSE_total(end) * avg_cost_S(end);
loglog(avg_cost_S, Ccost_rel ./ avg_cost_S, '--','LineWidth',1.2);

xlabel('Expected cost = S * E[cost per estimator]');
ylabel('Total relative MSE');
title(sprintf('Relative MSE vs expected cost (t-Student unbiased, \\nu=%.1e)', nu_large));
legend('rel MSE','C / cost','Location','southwest');


%% ===== Test: MSE of averaged t-Student unbiased estimators vs S and vs cost =====
clear; clc; rng(42);
%% 1) Model & synthetic data (Gaussian SSM, same as before)
T          = 30;
theta_true = 0.95;
q_true     = 0.2^2;
r_true     = 0.3^2;
S0_true    = q_true/(1 - theta_true^2);
% Latent AR(1)
x = zeros(1,T); 
x(1) = sqrt(S0_true)*randn;
for t = 2:T
    x(t) = theta_true*x(t-1) + sqrt(q_true)*randn;
end

% Gaussian observations (true model)
y = x + sqrt(r_true)*randn(1,T);

%% 2) Analytical Gaussian score (ground truth for comparison)
[sc_gauss, ~] = score_gaussian_ssm(y, theta_true, q_true, r_true, S0_true);
g_true = [sc_gauss.dtheta; sc_gauss.dq; sc_gauss.dr];  % 3×1

fprintf('Analytical Gaussian score:\n');
fprintf('  dtheta = %+ .4e\n', sc_gauss.dtheta);
fprintf('  dq     = %+ .4e\n', sc_gauss.dq);
fprintf('  dr     = %+ .4e\n', sc_gauss.dr);

%% 3) PF / PG ingredients with t-Student observations (large nu)

% Initial distribution for PF/CPF
in_pars.mu    = 0;
in_pars.S0 = S0_true;
in_dist_samp  = @(p,N,M) reshape( ...
                        p.mu + sqrt(p.S0)*randn(1,N*M), ...
                        1, N, M);

% Transition
trans_pars.theta = theta_true;
trans_pars.q     = q_true;
trans_pars.sig   = sqrt(q_true);
trans_dist_samp  = @(Xprev,p,t) ...
    p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

% t-Student observation: large nu, scale = sqrt(r_true)
nu_large         = 1e5;
obs_pars_t.v     = nu_large;
obs_pars_t.sigma = sqrt(r_true);

% t-Student log-likelihood for PF/CPF
g_t = @(yt,Xt,p,t) log_g_t_stud(yt,Xt,p,t);   % must return 1×N×M

% Transition log-density for backward CPF/PG
trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - theta_true*X_prev).^2)/q_true ...
    - 0.5*log(2*pi*q_true);   % 1×N

%% 4) Unbiased estimator configuration (Rhee-Glynn levels)

B0    = 1;              % base number of PG links for level 0
eLes  = 0:10;             % example levels {0,1}; extend if desired
l_raw = (eLes+4).*log(eLes+4).^2 ./ 2.^eLes;
l_dist = l_raw / sum(l_raw);   % normalize to sum to 1

% PG / PF configuration inside each unbiased estimator
N      = 10;             % particles for PG
M      = 2;              % parallel chains per PG call
cpf_choice = "cpf";      % or "cpf_parallel"
traj_mode  = "backward"; % recommended with trans_logpdf
store_parts = false;
store_anc   = false;
store_logw  = false;
N_init      = N;
init_mode   = "weighted";

%% 5) Outer MSE test over S (number of i.i.d. unbiased estimators averaged)

% S-values (number of unbiased estimators per average)
S_vals = 2.^(3:7);       % e.g. S in {1,2,4,8,...,64}
KS     = numel(S_vals);

% Number of independent "averaging experiments" per S (to estimate MSE)
Krep   = 100;
% Storage
err_comp  = zeros(Krep, KS, 3);   % component-wise errors [dtheta,dq,dr]
err_total = zeros(Krep, KS);      % ||error||^2

% Theoretical expected cost per unbiased estimator:
% cost(L) = Bs(end) = B0 * 2^L, and L ~ (eLes, l_dist)
expected_cost_per_est = sum( B0 * 2.^eLes .* l_dist );
avg_cost_S = S_vals * expected_cost_per_est;   % expected cost for averaging S estimators

seed_base = 12345;

fprintf('\nRunning MSE test for t-Student unbiased estimator (nu = %.1e)...\n', nu_large);
tic;
for iS = 1:KS
    S = S_vals(iS);

    for k = 1:Krep

        % Collect S independent unbiased estimators of the 3×1 score
        ests = zeros(3, S);
        for s = 1:S
            seed0 = seed_base + 100000*iS + 1000*k + s;  % decorrelate seeds
            ests(:,s) = pg_unbiased_score_tstud( ...
                y, T, N, M, ...
                in_dist_samp, in_pars, ...
                trans_dist_samp, trans_pars, ...
                g_t, obs_pars_t, ...
                trans_logpdf, ...
                B0, eLes, l_dist, ...
                seed0, ...
                cpf_choice, traj_mode, ...
                store_parts, store_anc, store_logw, ...
                N_init, init_mode);
        end

        % Average over S estimators
        g_hat = mean(ests, 2);           % 3×1

        % Error vs analytical Gaussian score
        err_vec = g_hat - g_true;        % 3×1
        err_comp(k, iS, :) = err_vec;    % store component-wise
        err_total(k, iS)   = sum(err_vec.^2);  % squared norm
    end

    fprintf('  S = %4d done.\n', S);
end
toc;

%% 6) Compute MSE, bias^2, variance vs S
% Component-wise MSE, bias^2, var for each S and each component
MSE_comp   = squeeze( mean(err_comp.^2, 1) );                    % KS × 3
bias_comp  = squeeze( mean(err_comp, 1) );                       % KS × 3
bias2_comp = bias_comp.^2;                                       % KS × 3
var_comp   = MSE_comp - bias2_comp;                              % KS × 3

% Total MSE, bias^2, var (sum over components)
MSE_total   = squeeze( mean(err_total, 1) );                     % KS × 1
mean_err    = squeeze( mean(err_comp, 1) ).';                    % 3×KS
bias2_total = sum(mean_err.^2, 1).';                             % KS × 1
var_total = sum(var_comp, 2);   % KS×1, much more stable
%% 6.5 (Pre-step) Relative MSE, bias^2 and variance
g_true2      = (g_true(:)).^2;        % 3x1
norm_g_true2 = sum(g_true2);          % scalar

% KS x 3 (implicit expansion works in recent MATLAB; otherwise use bsxfun)
rel_MSE_comp   = MSE_comp   ./ (g_true2.'); 
rel_bias2_comp = bias2_comp ./ (g_true2.');
rel_var_comp   = var_comp   ./ (g_true2.');

% KS x 1
rel_MSE_total   = MSE_total   / norm_g_true2;
rel_bias2_total = bias2_total / norm_g_true2;
rel_var_total = rel_MSE_total.'-rel_bias2_total;   % KS×1, much more stable

%% 7) Plot: MSE vs S with 1/S reference (total and components)
comp_names = {'d\theta','dq','dr'};
figure;
for j = 1:3
    subplot(2,2,j);
    loglog(S_vals, MSE_comp(:,j), 'o-','LineWidth',1.5); hold on; grid on;
    loglog(S_vals, bias2_comp(:,j), 's--','LineWidth',1.0);
    loglog(S_vals, var_comp(:,j),  'd--','LineWidth',1.0);

    % 1/S reference (calibrate at largest S)
    Cj = MSE_comp(end,j) * S_vals(end);
    loglog(S_vals, Cj ./ S_vals, ':','LineWidth',1.2);

    xlabel('S (number of unbiased estimators averaged)');
    ylabel(['MSE(',comp_names{j},')']);
    title(['Component: ',comp_names{j}]);
    legend('MSE','bias^2','var','C/S','Location','southwest');
end
subplot(2,2,4);
loglog(S_vals, MSE_total, 'o-','LineWidth',1.5); hold on; grid on;
loglog(S_vals, bias2_total, 's--','LineWidth',1.0);
loglog(S_vals, var_total,  'd--','LineWidth',1.0);
Ctot = MSE_total(end) * S_vals(end);
loglog(S_vals, Ctot ./ S_vals, ':','LineWidth',1.2);
xlabel('S'); ylabel('Total MSE');
title('Total score MSE (sum of components)');
legend('MSE','bias^2','var','C/S','Location','southwest');
sgtitle(sprintf('t-Student unbiased score (nu=%.1e) vs Gaussian analytical score', nu_large));

%% 8) Plot: MSE vs expected cost (total)
figure;
loglog(avg_cost_S, MSE_total, 'o-','LineWidth',1.6); hold on; grid on;
Ccost = MSE_total(end) * avg_cost_S(end);
loglog(avg_cost_S, Ccost ./ avg_cost_S, '--','LineWidth',1.2);
xlabel('Expected cost = S * E[cost per estimator]');
ylabel('Total MSE');
title(sprintf('MSE vs expected cost (t-Student unbiased, \\nu=%.1e)', nu_large));
legend('MSE','C / cost','Location','southwest');
%% 9) Plot: relative MSE vs S with 1/S reference (total and components)
comp_names = {'d\mu','d(\Sigma^2)','d(\sigma^2)'};
figure;
for j = 1:3
    subplot(2,2,j);
    loglog(S_vals, rel_MSE_comp(:,j), 'o-','LineWidth',1.5); hold on; grid on;
    loglog(S_vals, rel_bias2_comp(:,j), 's--','LineWidth',1.0);
    loglog(S_vals, rel_var_comp(:,j),  'd--','LineWidth',1.0);

    % 1/S reference (calibrate at largest S) for relative MSE
    Cj_rel = rel_MSE_comp(end,j) * S_vals(end);
    loglog(S_vals, Cj_rel ./ S_vals, ':','LineWidth',1.2);

    xlabel('S (number of unbiased estimators averaged)');
    ylabel(['rel. MSE(',comp_names{j},')']);
    title(['Component: ',comp_names{j}]);
    legend('rel MSE','rel bias^2','rel var','C/S','Location','southwest');
end

subplot(2,2,4);
loglog(S_vals, rel_MSE_total, 'o-','LineWidth',1.5); hold on; grid on;
loglog(S_vals, rel_bias2_total, 's--','LineWidth',1.0);
loglog(S_vals, rel_var_total,  'd--','LineWidth',1.0);

Ctot_rel = rel_MSE_total(end) * S_vals(end);
loglog(S_vals, Ctot_rel ./ S_vals, ':','LineWidth',1.2);

xlabel('S');
ylabel('Total relative MSE');
title('Total score relative MSE (sum of components)');
legend('rel MSE','rel bias^2','rel var','C/S','Location','southwest');

sgtitle(sprintf('Relative MSE vs S (t-Student unbiased score, \\nu=%.1e)', nu_large));
%% 9.5) Separate plots of relative MSE components and total (export as PDF)

fig_path = ['/Users/alvarem/Library/CloudStorage/GoogleDrive-miguelangel.alvarezballesteros@kaust.edu.sa/', ...
            'Other computers/My MacBook Pro/MEGA/0KAUST/0CSO_project/Figs'];

% ensure directory exists
if ~exist(fig_path, 'dir')
    mkdir(fig_path);
end

comp_names = {'d\mu','d(\Sigma^2)','d(\sigma^2)'};
file_tag   = {'relMSEvsS_tstud_large_v_dmu','relMSEvsS_tstud_large_v_d1Sigma2'...
    ,'relMSEvsS_tstud_large_v_d2sigma2'};   % for filenames

% --- 9.5.1 Component-wise figures ---
for j = 1:3
    f = figure;
    loglog(S_vals, rel_MSE_comp(:,j), 'o-','LineWidth',3.5); hold on; grid on;
    loglog(S_vals, rel_bias2_comp(:,j), 's--','LineWidth',3.0);
    loglog(S_vals, rel_var_comp(:,j),  'd--','LineWidth',3.0);

    % 1/S reference (calibrate at largest S) for relative MSE
    Cj_rel = rel_MSE_comp(end,j) * S_vals(end);
    loglog(S_vals, Cj_rel ./ S_vals, ':','LineWidth',3.2);

    xlabel('S (number of unbiased estimators averaged)');
    ylabel(['rel. MSE(',comp_names{j},')']);
    %ylabel(['rel. MSE(',comp_names{j},')'], 'Interpreter','latex');
    title(['Component: ',comp_names{j}, ...
           sprintf(' (\\nu=%.1e)', nu_large)]);
    legend('rel MSE','rel bias^2','rel var','C/S', ...
           'Location','southwest');

    set(gca, 'FontSize', 25);

    % export to PDF
    out_file = fullfile(fig_path, ...
        sprintf('relMSE_component_%s_nu_%1.1e.pdf', file_tag{j}, nu_large));
%    exportgraphics(f, out_file, 'ContentType','vector');
end

% --- 9.5.2 Total relative MSE figure ---
f_tot = figure;
loglog(S_vals, rel_MSE_total, 'o-','LineWidth',3.5); hold on; grid on;
loglog(S_vals, rel_bias2_total, 's--','LineWidth',3.0);
loglog(S_vals, rel_var_total,  'd--','LineWidth',3.0);

Ctot_rel = rel_MSE_total(end) * S_vals(end);
loglog(S_vals, Ctot_rel ./ S_vals, ':','LineWidth',3.2);

xlabel('S (number of unbiased estimators averaged)', 'Interpreter','latex');
ylabel('Total relative MSE', 'Interpreter','latex');
title(sprintf('Total score relative MSE (\\nu=%.1e)', nu_large));
legend('rel MSE','rel bias^2','rel var','C/S', ...
       'Location','southwest','Interpreter','latex');

set(gca, 'FontSize', 25);

% export total figure
out_file_tot = fullfile(fig_path, ...
    sprintf('relMSE_total_nu_%1.1e.pdf', nu_large));
% exportgraphics(f_tot, out_file_tot, 'ContentType','vector');
%% 10) Plot: relative MSE vs expected cost (total)
figure;
loglog(avg_cost_S, rel_MSE_total, 'o-','LineWidth',1.6); hold on; grid on;

Ccost_rel = rel_MSE_total(end) * avg_cost_S(end);
loglog(avg_cost_S, Ccost_rel ./ avg_cost_S, '--','LineWidth',1.2);

xlabel('Expected cost = S * E[cost per estimator]');
ylabel('Total relative MSE');
title(sprintf('Relative MSE vs expected cost (t-Student unbiased, \\nu=%.1e)', nu_large));
legend('rel MSE','C / cost','Location','southwest');

%% ============================================================
%  Test: PG + t-Student obs (large nu) vs analytical Gaussian score
%  - Latent AR(1) with Gaussian noise
%  - Observations: Gaussian (for ground-truth Gaussian model)
%  - PG model: same AR(1) but t-Student likelihood with large nu
%  - Score from PG paths (t-based) vs analytical Gaussian score
% ============================================================

clear; clc; rng(123);

%% 1) Model & synthetic data (Gaussian observations)
T          = 20;
theta_true = 0.95;
q_true     = 0.2^2;
r_true     = 0.3^2;
S0_true    = q_true / (1 - theta_true^2);
sigma_true = sqrt(r_true);

% latent AR(1)
x = zeros(1,T);
x(1) = sqrt(S0_true)*randn;
for t = 2:T 
    x(t) = theta_true*x(t-1) + sqrt(q_true)*randn;
end

% Gaussian observations (this is the "true" model for score_gaussian_ssm)
y = x + sigma_true*randn(1,T);

%% 2) PF / CPF / PG definitions with t-Student obs (large nu)

% Initial distribution: N(0, S0_true)
init_pars.mu    = 0;
init_pars.S0 = S0_true;
in_dist_samp    = @(p, N, M) reshape( ...
                        p.mu + sqrt(p.S0)*randn(1, N*M), ...
                        1, N, M);

% Transition: x_t = theta * x_{t-1} + sqrt(q)*epsilon
trans_pars.theta = theta_true;
trans_pars.q     = q_true;
trans_pars.sig   = sqrt(q_true);
trans_dist_samp  = @(Xprev, p, t) ...
    p.theta * Xprev + p.sig * randn(size(Xprev), 'like', Xprev);

% t-Student observation parameters with large nu
nu_large         = 1e7;
obs_pars_t.v     = nu_large;
obs_pars_t.sigma = sigma_true;

% t-Student log-likelihood for PF/CPF:
% log_g_t_stud must return 1×N×M
g_t = @(yt, Xt, p, t) log_g_t_stud(yt, Xt, p, t);

% Transition log-density for backward simulation in CPF/PG (Gaussian AR(1))
trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - theta_true*X_prev).^2)/q_true ...
    - 0.5*log(2*pi*q_true);   % 1×N

%% 3) Particle Gibbs settings

N      = 50;          % particles
M      = 2;           % CPF chains per PG iteration
B      = 10000;         % PG links (so chain length = B+1)
seed0  = 321;

cpf_choice = "cpf";       % use serial cpf (simpler for timing)
traj_mode  = "backward";  % backward simulation

store_parts = false;
store_anc   = false;
store_logw  = false;
N_init      = N;
init_mode   = "weighted";

%% 4) Run Particle Gibbs with t-Student obs (large nu)

out_pg = pgibbs_run_init_pfmean( ...
    y, T, N, M, B, ...
    in_dist_samp, init_pars, ...
    trans_dist_samp, trans_pars, ...
    g_t, obs_pars_t, ...
    seed0, ...
    cpf_choice, traj_mode, trans_logpdf, ...
    store_parts, store_anc, store_logw, ...
    N_init, init_mode);

% out_pg.X_paths: 1×T×M×(B+1)
X_paths_all = out_pg.X_paths;
[~, ~, M_, Bp1] = size(X_paths_all);
P = M_ * Bp1;

% Flatten chains and filters into a single "path index" dimension P
X_paths = reshape(X_paths_all, 1, T, P);   % 1×T×P

fprintf('PG paths flattened: P = %d (M=%d, B+1=%d)\n', P, M_, Bp1);

%% 5) Analytical Gaussian score (ground truth)

[sc_gauss, ~] = score_gaussian_ssm(y, theta_true, q_true, r_true, S0_true);

% Collect the components you want to compare:
% Transition parameters: dtheta, dq
g_anal_trans = [sc_gauss.dtheta; sc_gauss.dq];   % 2×1

% (Optionally) initial variance:
g_anal_init  = sc_gauss.dS0;                     % scalar

fprintf('\nAnalytical Gaussian score:\n');
fprintf('  dtheta = %+ .4e\n', sc_gauss.dtheta);
fprintf('  dq     = %+ .4e\n', sc_gauss.dq);
fprintf('  dr     = %+ .4e\n', sc_gauss.dr);
fprintf('  dS0    = %+ .4e\n', sc_gauss.dS0);

%% 6) Score from PG paths using t-based vectorized score
% Here score_t_from_paths_vectorized uses:
%  - Gaussian init & transition gradients
%  - t-Student obs gradient wrt log(sigma) with nu_large
S_t = score_t_from_paths_vectorized(y, X_paths, init_pars, trans_pars, obs_pars_t);

% Transition components (2×1): should be close to [dtheta; dq]
g_pg_trans = S_t.avg_trans;     % 2×1
g_pg_init  = S_t.avg_init;      % scalar

g_pg_obs   = S_t.avg_obs;       % scalar (wrt log sigma, not directly comparable to dr)
% Convert avg_obs (dℓ/dlogσ) to an approximate dℓ/dr using r_true
dr_pg_from_t = g_pg_obs / (2 * r_true);

fprintf('\nPG+t (nu = %.1e) Monte Carlo score from paths:\n', nu_large);
fprintf('  avg_init      = %+ .4e\n', g_pg_init);
fprintf('  avg_trans(1)  = dtheta ≈ %+ .4e\n', g_pg_trans(1));
fprintf('  avg_trans(2)  = dq     ≈ %+ .4e\n', g_pg_trans(2));
fprintf('  avg_obs (logσ)          = %+ .4e\n', g_pg_obs);

%% 7) Quantitative comparison (transition & init)

% Transition difference
diff_trans_vec = g_pg_trans - g_anal_trans;         % 2×1
diff_trans_vec = squeeze(diff_trans_vec);
norm_diff_trans = norm(diff_trans_vec);
rel_diff_trans  = norm_diff_trans / max(1e-12, norm(g_anal_trans));

% Initial variance difference (if grad_log_p1_gauss matches dS0)
diff_init = g_pg_init - g_anal_init;
rel_diff_init = abs(diff_init) / max(1e-12, abs(g_anal_init));

fprintf('\nDifferences (PG+t large nu vs analytical Gaussian):\n');
fprintf('  Transition components:\n');
fprintf('    dtheta:  MC = %+ .4e, Analytic = %+ .4e, diff = %+ .3e\n', ...
        g_pg_trans(1), g_anal_trans(1), diff_trans_vec(1));
fprintf('    dq    :  MC = %+ .4e, Analytic = %+ .4e, diff = %+ .3e\n', ...
        g_pg_trans(2), g_anal_trans(2), diff_trans_vec(2));
fprintf('  ||diff_trans||2          = %.3e\n', norm_diff_trans);
fprintf('  relative ||diff_trans||2 = %.3e\n', rel_diff_trans);

fprintf('\n  Initial variance component (if applicable):\n');
fprintf('    MC = %+ .4e, Analytic = %+ .4e, diff = %+ .3e, rel = %.3e\n', ...
        g_pg_init, g_anal_init, diff_init, rel_diff_init);

dr_anal = sc_gauss.dr;

diff_dr     = dr_pg_from_t - dr_anal;
rel_diff_dr = abs(diff_dr) / max(1e-12, abs(dr_anal));

fprintf('\nObservation variance score comparison (t-large-nu vs Gaussian):\n');
fprintf('  from t (via logσ -> r):   %+ .4e\n', dr_pg_from_t);
fprintf('  analytical Gaussian dr:  %+ .4e\n', dr_anal);
fprintf('  diff = %+ .3e,  rel diff = %.3e\n', diff_dr, rel_diff_dr);

%%


%% ============================================================
%  Test: t-Student observations, PG paths, score comparison
%  - Simulate AR(1) state
%  - Generate t-Student observations
%  - Run particle Gibbs
%  - Compare score_t_from_paths vs score_t_from_paths_vectorized
% ============================================================

clear; clc; rng(10);
%% 1) Model & synthetic data (latent Gaussian AR(1), t obs)
T          = 20;
theta_true = 0.95;
q_true     = 0.2^2;
S0_true    = q_true/(1 - theta_true^2);

v_true     = 5;        % t degrees of freedom
sigma_true = 0.3;      % t scale

% latent state
x = zeros(1,T);
x(1) = sqrt(S0_true)*randn;
for t = 2:T
    x(t) = theta_true*x(t-1) + sqrt(q_true)*randn;
end

% t-Student noise
t_noise = trnd(v_true, 1, T);      % 1×T Student-t(v)
y       = x + sigma_true * t_noise;

%% 2) PF / CPF / PG definitions

% Initial distribution: N(0, S0_true)
init_pars.mu    = 0;
init_pars.Sigma = S0_true;
init_pars_0.mu    = 0;
init_pars_0.S0 = S0_true;
in_dist_samp    = @(p, N, M) reshape( ...
                        p.mu + sqrt(p.Sigma)*randn(1, N*M), ...
                        1, N, M);

% Transition: x_t = theta * x_{t-1} + sqrt(q) * eps
trans_pars.theta = theta_true;
trans_pars.q     = q_true;
trans_pars.sig   = sqrt(q_true);
trans_dist_samp  = @(Xprev, p, t) ...
    p.theta * Xprev + p.sig * randn(size(Xprev), 'like', Xprev);

% t-Student observation parameters
obs_pars.v     = v_true;
obs_pars.sigma = sigma_true;

% t-Student log-likelihood for PF/CPF:
%   log_g_t_stud(yt, Xt, obs_pars, t) must return 1×N×M
g_t = @(yt, Xt, p, t) log_g_t_stud(yt, Xt, p, t);

% Transition log-density for backward simulation in CPF/PG (Gaussian)
trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - theta_true*X_prev).^2)/q_true ...
    - 0.5*log(2*pi*q_true);   % 1×N

%% 3) Particle Gibbs settings
N      = 10;          % particles
M      = 2;           % parallel CPF chains per PG iteration
B      = 40;          % PG links (so chain length B+1)
seed0  = 1234;

cpf_choice = "cpf";       % use serial cpf here (no need for parallel)
traj_mode  = "backward";  % or "ancestors"

store_parts = false;
store_anc   = false;
store_logw  = false;
N_init      = N;
init_mode   = "weighted";

%% 4) Run Particle Gibbs with t-Student observations

out_pg = pgibbs_run_init_pfmean( ...
    y, T, N, M, B, ...
    in_dist_samp, init_pars, ...
    trans_dist_samp, trans_pars, ...
    g_t, obs_pars, ...
    seed0, ...
    cpf_choice, traj_mode, trans_logpdf, ...
    store_parts, store_anc, store_logw, ...
    N_init, init_mode);

% out_pg.X_paths should be: 1×T×M×(B+1)
X_paths_all = out_pg.X_paths;   % 1×T×M×(B+1)

% Flatten M and chain dimension into a single "P" (number of paths)
[~, ~, M_, Bp1] = size(X_paths_all);
P = M_ * Bp1;
X_paths = reshape(X_paths_all, 1, T, P);   % 1×T×P

fprintf('Flattened PG paths: P = %d (M=%d, B+1=%d)\n', P, M_, Bp1);

%% 5) Compute scores: loop vs vectorized

% Loop version (score_t_from_paths)
S_loop = score_t_from_paths(y, X_paths, init_pars_0, trans_pars, obs_pars);

% Vectorized version (score_t_from_paths_vectorized)
S_vec  = score_t_from_paths_vectorized(y, X_paths, init_pars_0, trans_pars, obs_pars);
%% 6) Numerical agreement (loop vs vectorized)

% Differences for initial (scalar), transition (2×1 vector), and obs (scalar)
diff_init      = abs(S_loop.avg_init - S_vec.avg_init);             % scalar
diff_trans_vec = S_loop.avg_trans - S_vec.avg_trans;                 % 2×1 vector

diff_trans     = norm(diff_trans_vec,2);                               % norm
diff_obs       = abs(S_loop.avg_obs - S_vec.avg_obs);                % scalar

size(S_vec.avg_obs)
size(diff_init)
size(diff_trans)



fprintf('\nNumerical differences (loop vs vectorized):\n');
fprintf('  |avg_init_loop - avg_init_vec|     = %.3e\n', diff_init);

fprintf('  Transition differences (component-wise):\n');
fprintf('      dtheta:     %.3e\n', diff_trans_vec(1));
fprintf('      dlogq:      %.3e\n', diff_trans_vec(2));
fprintf('  Norm of transition difference:       %.3e\n', diff_trans);

fprintf('  |avg_obs_loop  - avg_obs_vec|       = %.3e\n', diff_obs);
%% 7) Timing comparison

R = 10000;   % repetitions

% Warm-up calls (ensures JIT compilation before timing)
score_t_from_paths(y, X_paths, init_pars_0, trans_pars, obs_pars);
score_t_from_paths_vectorized(y, X_paths, init_pars_0, trans_pars, obs_pars);

% --- Loop version timing ---

size(X_paths)
tic;

for r = 1:R
    S1 = score_t_from_paths(y, X_paths, init_pars_0, trans_pars, obs_pars); %#ok<NASGU>
end
t_loop = toc / R;

% --- Vectorized version timing ---
tic;
for r = 1:R
    S2 = score_t_from_paths_vectorized(y, X_paths, init_pars_0, trans_pars, obs_pars); %#ok<NASGU>
end
t_vec = toc / R;

fprintf('\nTiming over %d repetitions:\n', R);
fprintf('  score_t_from_paths (loop)           : %.4f s per call\n', t_loop);
fprintf('  score_t_from_paths_vectorized       : %.4f s per call\n', t_vec);
fprintf('  Speedup (loop / vectorized)         : %.2f×\n', t_loop / t_vec);
%% ==============================================================
%   Variance comparison of unbiased estimator:
%        base level (ℓ=0) vs second level (ℓ=1)
% ==============================================================

clear; clc;

%% 1. Model and data
T     = 10;
theta = 0.95;
q     = 0.2^2;
r     = 0.3^2;
S0    = q/(1-theta^2);

rng(2);
x = zeros(1,T); x(1)=sqrt(S0)*randn;
for t=2:T, x(t) = theta*x(t-1)+sqrt(q)*randn; end
y = x + sqrt(r)*randn(1,T);

%% 2. PF/PG parameters
N      = 10;
M      = 2;
seed0  = 12345;
traj_mode = "backward";

in_pars.mu = 0;
in_pars.Sigma = S0;
in_dist = @(p,N_,M_) reshape(p.mu+sqrt(p.Sigma)*randn(1,N_*M_),1,N_,M_);

tr_pars.theta = theta;
tr_pars.q     = q;
tr_pars.sig   = sqrt(q);
trans = @(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

g_pars.R = r;
g = @(yt,Xt,p,t) reshape( -0.5*((yt-Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
                          1, size(Xt,2), size(Xt,3));

trans_logpdf = @(x_next,X_prev,pars,t) ...
    -0.5*((x_next-theta*X_prev).^2)/q - 0.5*log(2*pi*q);

%% 3. Set level distributions
%  ℓ=0 only
l_dist0 = [1, 0];

%  ℓ=1 only
l_dist1 = [0, 1];

% Base B0
B0 = 40;            % same as your test
B1 = B0*2;          % next level

%% 4. Replications
K = 100;
unb0 = zeros(K, 3);   % unbiased estimates (dr, dtheta, dq)
unb1 = zeros(K, 3);

fprintf("Running %d replications...\n", K);

for i = 1:K
    seed_i = 1000 + i*17;
    % ---------------------------
    % ℓ = 0 (base level)
    % ---------------------------
    B = B0;
    out0 = pgibbs_run_init_pfmean( ...
        y,T,N,M,B, ...
        in_dist,in_pars, ...
        trans,tr_pars, ...
        g,g_pars, ...
        seed_i, ...
        "cpf", ...
        traj_mode,trans_logpdf, ...
        false,false,false, ...
        N,"weighted");

    S0_est = score_gaussian_from_paths_vectorized( ...
                y, out0.X_paths, theta, q, r, S0,[],[], true);

    g0 = [S0_est.avg_trans(1); S0_est.avg_trans(2); S0_est.avg_obs];
    unb0(i,:) = (B0 * g0).' / (B0 * l_dist0(1));
    % ---------------------------
    % ℓ = 1 (increment level)
    % ---------------------------
    % Compute level 0 and level 1 contributions
    B_list = [B0, B1];
    g_levels = zeros(2,3);  % store g0, g1

    for ell = 1:2
        B = B_list(ell);
        tic;
        out = pgibbs_run_init_pfmean( ...
            y,T,N,M,B, ...
            in_dist,in_pars, ...
            trans,tr_pars, ...
            g,g_pars, ...
            seed_i + 1000*ell, ...
            "cpf", ...
            traj_mode,trans_logpdf, ...
            false,false,false, ...
            N,"weighted");
        Sest = score_gaussian_from_paths_vectorized( ...
                    y, out.X_paths, theta, q, r, S0,[],[], true);

        g_levels(ell,:) = (B * [Sest.avg_trans(1), Sest.avg_trans(2), Sest.avg_obs]);
    end

    % ℓ=1 unbiased increment
    numer = g_levels(2,:) - g_levels(1,:);
    unb1(i,:) = numer / (B1 * l_dist1(2));
end

%% 5. Variance comparison
var0 = var(unb0);      % base level variance
var1 = var(unb1);      % second level variance
disp("Variance of unbiased estimator (base level):");
disp(var0);
disp("Variance of unbiased estimator (second level):");
disp(var1);

%% 6. Simple plots
figure;
bar([var0; var1]');
set(gca,'XTickLabel',{'dr','dtheta','dq'});
legend('Base level','Second level');
title('Variance comparison of unbiased estimator levels');


%% ===== Timing test: cpf vs cpf_parallel as a function of M =====
clear; clc;
ps = parallel.Settings;
ps.Pool.AutoCreate = false;
%% 1) Model & synthetic data (same LGSSM as before)
T          = 1000;
theta_true = 0.95;
q_true     = 0.2^2;
r_true     = 0.3^2;
S0_true    = q_true/(1 - theta_true^2);
rng(42);
x_true = zeros(1,T); 
x_true(1) = sqrt(S0_true)*randn;
for t = 2:T
    x_true(t) = theta_true*x_true(t-1) + sqrt(q_true)*randn;
end
y = x_true + sqrt(r_true)*randn(1,T);

d_x = 1; d_y = 1;

%% 2) User functions (in_dist, trans, g, trans_logpdf)
in_pars.mu    = 0;
in_pars.Sigma = S0_true;
in_dist = @(p,N,M) reshape( ...
                    p.mu + sqrt(p.Sigma)*randn(1,N*M), ...
                    1, N, M);

tr_pars.theta = theta_true;
tr_pars.q     = q_true;
tr_pars.sig   = sqrt(q_true);
trans = @(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev), 'like', Xprev);

g_pars.R = r_true;
g = @(yt,Xt,p,t) reshape( ...
            -0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
            1, size(Xt,2), size(Xt,3));

trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - theta_true*X_prev).^2)/q_true - 0.5*log(2*pi*q_true);

%% 3) CPF settings for timing
N      = 10;               % # particles per filter
Ms     = 5000*[10];      % different numbers of filters
n_rep  = 2;                 % repetitions per M to smooth noise
seed0  = 12345;
traj_mode = "backward";     % or "ancestors"
store_parts = false;
store_anc   = false;
store_logw  = false;

% OPTIONAL: start a parallel pool if you want cpf_parallel to actually run in parallel
%parpool ('local', 4);   % adjust number of workers

times_cpf       = zeros(numel(Ms),1);
times_cpf_par   = zeros(numel(Ms),1);

%% 4) Timing loop
%delete(gcp('nocreate'))
%parpool('local', 3)
for i = 1:numel(Ms)
    M = Ms(i);
    % Build a reference path x_ref of size d_x × T × M
    x_ref = reshape(x_true, d_x, T, 1);  % 1 × T × 1
    x_ref = repmat(x_ref, 1, 1, M);      % 1 × T × M
    
    t_cpf_acc     = 0;
    t_cpf_par_acc = 0;

    for r = 1:n_rep
        seed_r = seed0 + 1000*r + 10*M;
        

        % ---- serial CPF ----
        tic;
        
        out1 = cpf( ...
            y, T, N, M, ...
            in_dist, in_pars, ...
            trans, tr_pars, ...
            g, g_pars, ...
            x_ref, seed_r, ...
            traj_mode, trans_logpdf, ...
            store_parts, store_anc, store_logw);
        t_cpf_acc = t_cpf_acc + toc;

        % ---- parallel CPF ----
        tic;
        out2 = cpf_parallel( ...
            y, T, N, M, ...
            in_dist, in_pars, ...
            trans, tr_pars, ...
            g, g_pars, ...
            x_ref, seed_r, ...
            traj_mode, trans_logpdf, ...
            store_parts, store_anc, store_logw);
        t_cpf_par_acc = t_cpf_par_acc + toc;
    end

    times_cpf(i)     = t_cpf_acc     / n_rep;
    times_cpf_par(i) = t_cpf_par_acc / n_rep;

    fprintf('M = %2d | cpf: %.4f s | cpf_parallel: %.4f s\n', ...
            M, times_cpf(i), times_cpf_par(i));
end

%% 5) Simple plot / table of timing results
results = table(Ms(:), times_cpf, times_cpf_par, ...
    'VariableNames', {'M', 'time_cpf', 'time_cpf_parallel'});
disp(results);
figure;
loglog(Ms, times_cpf, 'o-', 'LineWidth', 1.6); hold on; grid on;
loglog(Ms, times_cpf_par, 's--', 'LineWidth', 1.6);
xlabel('M (number of filters)');
ylabel('Average wall time (seconds)');
title(sprintf('CPF vs CPF\\_PARALLEL timing (N=%d, T=%d, reps=%d)', N, T, n_rep));
legend('cpf (serial)','cpf\_parallel','Location','northwest');


%%
clear; clc;
%%
%% ===== Test: MSE of averaged unbiased estimators vs S and vs cost =====
clear; clc;
delete(gcp('nocreate'))
%% 1) Model & data (same as before)
T     = 10;
theta = 0.95;
q     = 0.2^2;
r     = 0.3^2;
S0    = q/(1-theta^2);

rng(42);
x = zeros(1,T); x(1) = sqrt(S0)*randn;
for t = 2:T, x(t) = theta*x(t-1) + sqrt(q)*randn; end
y = x + sqrt(r)*randn(1,T);

%% 2) PF/PG functions (same as before)
in_pars.mu    = 0; 
in_pars.Sigma = S0;
in_dist = @(p,N,M) reshape(p.mu + sqrt(p.Sigma)*randn(1,N*M), 1, N, M);

tr_pars.theta = theta; 
tr_pars.q     = q; 
tr_pars.sig   = sqrt(q);
trans = @(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

g_pars.R = r;
g = @(yt,Xt,p,t) reshape(-0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
                         1, size(Xt,2), size(Xt,3));

trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next-theta*X_prev).^2)/q - 0.5*log(2*pi*q);

%% 3) Analytical score
[sc_anal, ~] = score_gaussian_ssm(y, theta, q, r, S0);
g_anal = [sc_anal.dtheta; sc_anal.dq; sc_anal.dr];   % 3×1

%% 4) Unbiased estimator configuration
N      = 10;
M      = 2;
cpf_choice = "cpf";
traj_mode  = "backward";
store_parts = false; store_anc = false; store_logw = false;
N_init  = N;
init_mode = "weighted";

% Level distribution and Bs (same style as your working code)
eLes   = 0:2;                          % levels {0,1} for now
l_dist = (eLes+4).*log(eLes+4).^2./2.^eLes;
l_dist = l_dist / sum(l_dist);
B0   = 1;
Bs   = B0 * 2.^(0:max(eLes));          % [20, 40] for levels 0,1

%% 5) Grid of S (number of i.i.d. unbiased estimators per average)
S_vals = 10*2.^(0:3);                     % S = 1,2,4,8,16,32
KS     = numel(S_vals);
K_reps = 20;                          % number of replicates for MSE estimate
% Optionally: store component-wise MSE if you like
%mse_comp = zeros(KS,3);

%% 6) Loop over S and replicates — now collect bias² and variance too
mse_total   = zeros(KS, 1);
bias2_total = zeros(KS, 1);
var_total   = zeros(KS, 1);
avg_cost    = zeros(KS, 1);

for is = 1:KS
    S_num = S_vals(is);

    G_bar_reps = zeros(3, K_reps);   % store the averaged estimator for each replicate
    cost_rep   = zeros(K_reps,1);

    for k = 1:K_reps
        G_bar    = zeros(3,1);   % running average for this replicate
        cost_sum = 0;

        for s = 1:S_num
            seed_s = 100000*is + 1000*k + s;

            [G_unb, info] = pg_unbiased_score_gauss( ...
                y, T, N, M, ...
                in_dist, in_pars, ...
                trans, tr_pars, ...
                g, g_pars, ...
                theta, q, r, S0, ...
                Bs, l_dist, seed_s, ...
                cpf_choice, traj_mode, trans_logpdf, ...
                store_parts, store_anc, store_logw, ...
                N_init, init_mode);

            G_bar    = G_bar + G_unb;
            cost_sum = cost_sum + info.total_cost;
        end

        G_bar = G_bar / S_num;

        G_bar_reps(:,k) = G_bar;
        cost_rep(k)     = cost_sum / S_num;
    end

    % Mean estimator over K_reps
    G_mean = mean(G_bar_reps, 2);

    % Bias vector
    bias_vec = G_mean - g_anal;

    % Bias squared (scalar)
    bias2_total(is) = norm(bias_vec, 2)^2;

    % Variance = E[||G - G_mean||^2]
    var_total(is) = mean(vecnorm(G_bar_reps - G_mean, 2, 1).^2);

    % MSE = bias² + variance
    mse_total(is) = bias2_total(is) + var_total(is);

    % Average cost
    avg_cost(is) = mean(cost_rep);
end

%% 7) Plots: MSE, bias², and variance vs S, and vs average cost

%% (A) MSE, Bias², Variance vs S
figure;
loglog(S_vals, mse_total, 'o-', 'LineWidth', 1.6); hold on;
loglog(S_vals, bias2_total, 's--', 'LineWidth', 1.6);
loglog(S_vals, var_total, 'd-.', 'LineWidth', 1.6);
% Reference line C/S
C_ref = mse_total(1) * S_vals(1);
loglog(S_vals, C_ref ./ S_vals, 'k:', 'LineWidth', 1.6);
grid on;
xlabel('S (# of i.i.d. unbiased draws)');
ylabel('Value');
title('Error decomposition vs S');
legend('MSE','Bias^2','Variance','C/S reference','Location','southwest');

%% (B) MSE, Bias², Variance vs average cost
figure;
loglog(avg_cost, mse_total, 'o-', 'LineWidth', 1.6); hold on;
loglog(avg_cost, bias2_total, 's--', 'LineWidth', 1.6);
loglog(avg_cost, var_total, 'd-.', 'LineWidth', 1.6);
% Reference line C2 / cost
C2_ref = mse_total(1) * avg_cost(1);
loglog(avg_cost, C2_ref ./ avg_cost, 'k:', 'LineWidth', 1.6);

grid on;
xlabel('Average cost per unbiased estimator');
ylabel('Value');
title('Error decomposition vs average cost');
legend('MSE','Bias^2','Variance','C/cost reference','Location','southwest');
%% 8) Compare MSE vs average cost
figure;
loglog(avg_cost, mse_total, 'o-','LineWidth',1.6); hold on; grid on;
% If you like, fit a line ~ C'' / cost:
C2_hat = mse_total(1) * avg_cost(1);
loglog(avg_cost, C2_hat ./ avg_cost, '--','LineWidth',1.2);
xlabel('Average cost per unbiased estimator');
ylabel('MSE(||Ḡ_S - g_{anal}||_2^2)');
title('MSE vs average cost');
legend('Empirical MSE','C'' \times cost^{-1}','Location','southwest');

% Unbiased estimator test
 %% 1. Generate data (true params)
T = 10;
theta_true = 0.95;
q_true     = 0.2^2;
r_true     = 0.3^2;
S0_true    = q_true/(1-theta_true^2);
rng(42);
x = zeros(1,T); x(1)=sqrt(S0_true)*randn;
for t=2:T, x(t)=theta_true*x(t-1)+sqrt(q_true)*randn; end
y = x + sqrt(r_true)*randn(1,T);

%% 2. SA settings
N=20; M=2; B=2;             % PF/PG settings
K=100;                         % SA iterations
seed0=123;

alpha = 0.2;                   % exponent (1/2, 1]
Gamma = [0.5; 20; 0];       % vector of step-sizes [theta,q,r]
Gamma=Gamma/T;
theta_max = 0.999;             % projection bound
theta0=0.8; q0=0.25^2; r0=r_true;%0.5^2;
theta=theta_true;
q=q_true;
r=r_true;
S0=S0_true;


% --- Build PF/CPF functions ---
in_pars.mu=0; in_pars.Sigma=S0;
in_dist=@(p,N_,M_) reshape(p.mu+sqrt(p.Sigma)*randn(1,N_*M_),1,N_,M_);
tr_pars.theta=theta; tr_pars.q=q; tr_pars.sig=sqrt(q);
trans=@(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);
g_pars.R=r;
g=@(yt,Xt,p,t) reshape(-0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R),1,size(Xt,2),size(Xt,3));
trans_logpdf=@(x_next,X_prev,pars,t) -0.5*((x_next-theta*X_prev).^2)/q - 0.5*log(2*pi*q);



K_ind=10000;
unb_ests=zeros(K_ind,3);
eLes=(0:1);
l_dist=(eLes+4).*log(eLes+4).^2./2.^eLes;
l_dist=l_dist/sum(l_dist);
for i=1:K_ind
    seed0=ceil(34256*abs(sin(i^2)));
    stream = RandStream('mrg32k3a', 'Seed', seed0 );
    j=draw_from_logw(log(l_dist));
    l=eLes(j);
    B0=20;
    Bs=B0*2.^(0:l);
    paths_num=B0*(1-2^(l+1))/(1-2);
    paths_sums=zeros(l+1,3);

    for n=1:(length(Bs))
    
        if n>1
        B=Bs(n)-Bs(n-1);
        else
        B=Bs(1);
        end

        out_pg=pgibbs_run_init_pfmean( ...
            y,T,N,M,B, ...
            in_dist,in_pars, ...
            trans,tr_pars, ...
            g,g_pars, ...
            seed0+1000*n, ...
            "cpf", ...
            "backward",trans_logpdf, ...
            false,false,false, ...
            N,"weighted");
    
    
        X_paths=out_pg.X_paths;
        S = score_gaussian_from_paths_vectorized(y,X_paths,theta,q,r,S0,[],[],true);
        g_vec = [S.avg_trans(1); S.avg_trans(2); S.avg_obs];  % [dtheta; dq; dr]
    
        paths_sums(n,:)=B*g_vec;
        
    
    end
    
    if l>0
        unb_ests(i,:) = (paths_sums(end,:) - sum(paths_sums(1:end-1,:),1)) / (Bs(end)*l_dist(j));
        
    else
        
        unb_ests(i,:) = paths_sums(1,:) /( Bs(1)*l_dist(j));
    end

    %unb_ests(i,:) = unb_ests(i,:).';   % if you really want column internally
end
%%
mean(unb_ests)
var(unb_ests)
%%
B=10000;
out_pg=pgibbs_run_init_pfmean( ...
            y,T,N,M,B, ...
            in_dist,in_pars, ...
            trans,tr_pars, ...
            g,g_pars, ...
            seed0+1000*n, ...
            "cpf", ...
            "backward",trans_logpdf, ...
            false,false,false, ...
            N,"weighted");
    
    
X_paths=out_pg.X_paths;
S = score_gaussian_from_paths_vectorized(y,X_paths,theta,q,r,S0,[],[],true);
g_vec = [S.avg_trans(1); S.avg_trans(2); S.avg_obs]  % [dtheta; dq; dr]


%%

trace = sa_pg_gauss_ssm(y, T, N, M, B, K, seed0, ...
    theta0, q0, r0, S0_true, Gamma, alpha, theta_max);



%%
% SA test

%% === SA for Gaussian SSM with PG-based stochastic score ===
clear; clc;

%% 1. Generate data (true params)
T = 30;
theta_true = 0.95;
q_true     = 0.2^2;
r_true     = 0.3^2;
S0_true    = q_true/(1-theta_true^2);

rng(42);
x = zeros(1,T); x(1)=sqrt(S0_true)*randn;
for t=2:T, x(t)=theta_true*x(t-1)+sqrt(q_true)*randn; end
y = x + sqrt(r_true)*randn(1,T);

%% 2. SA settings
N=10; M=2; B=500;             % PF/PG settings
K=50;                         % SA iterations
seed0=123;

alpha = 0.1;                   % exponent (1/2, 1]
Gamma = [5; 30; 0];       % vector of step-sizes [theta,q,r]
Gamma=Gamma/T;
theta_max = 0.999;             % projection bound
theta0=0.8; q0=0.25^2; r0=r_true;%0.5^2;
%delete(gcp('nocreate'))

trace = sa_pg_gauss_ssm(y, T, N, M, B, K, seed0, ...
    theta0, q0, r0, S0_true, Gamma, alpha, theta_max);

%% 3. 2D analytical gradient field (for validation)
n_grid = 300;
theta_vals = linspace(0.55, 0.7, n_grid);
q_vals = linspace(0.05,0.065, n_grid);  % fix r at r_true

[TH, QQ] = meshgrid(theta_vals, q_vals);
GRAD_T = zeros(size(TH));
GRAD_Q = zeros(size(QQ));

for i=1:n_grid
    for j=1:n_grid
        s = score_gaussian_ssm(y, TH(i,j), QQ(i,j), r_true, S0_true);
        GRAD_T(i,j) = s.dtheta;
        GRAD_Q(i,j) = s.dq;
    end
end

%% 4. Plot SA trajectory and analytical gradient field
figure;
contour(TH, QQ, sqrt(GRAD_T.^2 + GRAD_Q.^2), 500); hold on;
quiver(TH, QQ, GRAD_T, GRAD_Q, 'k'); grid on;
plot(trace.theta, trace.q, 'r-o', 'LineWidth',1.5, 'MarkerSize',4);
xlabel('\theta'); ylabel('q');
title('Analytical gradient field and SA trajectory (r fixed)');
legend('||score||','score field','SA path');
%% 4.5) Gradient field in evolved variables: (theta, log q) with r fixed

% === CHOOSE HOW TO SET LIMITS ============================================
use_manual_limits = true;   % <-- set to false to use automatic limits

if use_manual_limits
    % ---- MANUAL LIMITS (EDIT THESE) ----
    th_lo = 0.6;
    th_hi = 0.76;
    lq_lo = -2.88;
    lq_hi = -2.7;
else
    % ---- AUTOMATIC LIMITS FROM SA PATH (as before, but slightly cleaned) ----
    th_lo = min(trace.theta(10:end)) - 0.002;
    th_hi = max(trace.theta(10:end)) + 0.002;

    lq_path = log(trace.q(:));
    lq_true = log(q_true);
    lq_lo   = min([lq_path; lq_true]) - 0.2;
    lq_hi   = max([lq_path; lq_true]) + 0.2;
end
% ========================================================================

n_grid2 = 150;                             % resolution
theta_grid = linspace(th_lo, th_hi, n_grid2);
lq_grid    = linspace(lq_lo, lq_hi, n_grid2);

[TH2, LQ2] = meshgrid(theta_grid, lq_grid);

% Analytical gradient components in (theta, log q) coordinates
GRAD_T2  = zeros(size(TH2));              % dℓ/dθ
GRAD_LQ2 = zeros(size(LQ2));              % dℓ/d(log q) = q * dℓ/dq

for i = 1:n_grid2
    for j = 1:n_grid2
        th = TH2(i,j);
        qg = exp(LQ2(i,j));
        s  = score_gaussian_ssm(y, th, qg, r_true, S0_true);
        GRAD_T2(i,j)  = s.dtheta;
        GRAD_LQ2(i,j) = qg * s.dq;        % chain rule
    end
end

% Plot: norm contours + quiver field + SA path in (theta, log q)
figure;
contour(TH2, LQ2, hypot(GRAD_T2, GRAD_LQ2), 400); hold on; grid on;
quiver(TH2, LQ2, GRAD_T2, GRAD_LQ2, 'k');
a=1;
plot(trace.theta(a:end), log(trace.q(a:end)), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4);

xlabel('\theta'); ylabel('log q');
title('Analytical gradient field in (\theta, log q) with r fixed');
legend('||score|| contours','score field','SA path','Location','best');
%% 5. Plot parameter evolution
figure;
subplot(3,1,1); plot(trace.theta,'-'); yline(theta_true,'--'); grid on; ylabel('\theta');
subplot(3,1,2); plot(trace.q,'-'); yline(q_true,'--'); grid on; ylabel('q');
subplot(3,1,3); plot(trace.r,'-'); yline(r_true,'--'); grid on; ylabel('r'); xlabel('iteration n');
sgtitle('SA parameter evolution');

% Gaussian test (SA to be)
%%
%% Now we check the error in terms of the lenght of the chain

%% ===== Score error vs B for Particle Gibbs (1D Gaussian SSM, chain averages) =====
clear; clc;

%% 1) Model & synthetic data
T     = 10;
theta = 0.95;
q     = 0.2^2;
r     = 0.3^2;
S0    = q/(1-theta^2);   % stationary initial var

rng(10);
x = zeros(1,T); 
x(1) = sqrt(S0)*randn;
for t = 2:T
    x(t) = theta*x(t-1) + sqrt(q)*randn;
end
y = x + sqrt(r)*randn(1,T);

%% 2) User functions (PF/CPF/PF-Gibbs)
in_pars.mu    = 0; 
in_pars.Sigma = S0;
in_dist = @(p,N,M) reshape(p.mu + sqrt(p.Sigma)*randn(1,N*M), 1, N, M);

tr_pars.theta = theta; 
tr_pars.q     = q; 
tr_pars.sig   = sqrt(q);
trans = @(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

g_pars.R = r;
g = @(yt,Xt,p,t) reshape( -0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
                          1, size(Xt,2), size(Xt,3));

trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - theta*X_prev).^2)/q - 0.5*log(2*pi*q);   % 1×N

%% 3) Analytical score (ground truth)
[sc_anal, ~] = score_gaussian_ssm(y, theta, q, r, S0);
% Order: [dr; dtheta; dq; dS0]
g_anal = [sc_anal.dr; sc_anal.dtheta; sc_anal.dq; sc_anal.dS0];  % [4×1]

%% 4) Experiment grid: B = 2^l and K independent PG chains per B
l_vals = 3:12;                 % e.g., B in {8,16,32,64,128,256}
Bs     = 2.^l_vals;
KB     = numel(Bs);

Krep   = 50;                  % number of independent chains per B

% PG settings
N      = 2;                  % particles
M      = 2;                   % parallel filters inside each CPF/PG (fixed)
seed0  = 123;
cpf_choice = "cpf_parallel";  % or "cpf"
traj_mode  = "backward";      % "ancestors" also possible

N_init    = N; 
init_mode = "weighted";

store_parts = false; 
store_anc   = false; 
store_logw  = false;

% Storage
err_norm   = zeros(Krep, KB);       % ||g_chain - g_anal||_2
err_comp   = zeros(Krep, KB, 4);    % component-wise abs error
time_pg    = zeros(Krep, KB);

%% 5) Helper: one PG chain, length B, return chain-average score
