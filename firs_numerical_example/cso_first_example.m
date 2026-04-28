
%% Test for the error of the unbiased score method.
clear; clc; rng(1);

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
K_SA  =4*100;                  
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
C=20;
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


%% save the data
% Choose a path and filename
data_path = ['/Users/alvarem/Library/CloudStorage/GoogleDrive-', ...
    'miguelangel.alvarezballesteros@kaust.edu.sa/Other computers/', ...
    'My MacBook Pro/MEGA/0KAUST/0CSO_project/Data'];
save(fullfile(data_path, 'trace_mix_all_SA_40000_larger_gamma_ksa.mat'), 'trace_mix_all');
%%
%file = fullfile(data_path, 'trace_mix_all_SA_40000_ksa.mat');
%file_l = load(file);   % load ONLY variable 'repro'
%trace_mix_all=file_l.trace_mix_all;
%t_m_a.theta

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
K=4*1000;
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
%%
data_path = ['/Users/alvarem/Library/CloudStorage/GoogleDrive-', ...
    'miguelangel.alvarezballesteros@kaust.edu.sa/Other computers/', ...
    'My MacBook Pro/MEGA/0KAUST/0CSO_project/Data'];
%save(fullfile(data_path, 'trace_mix_lik_all_biased_path_truth.mat'), 'trace_mix_lik_all');


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
fig_path = ['/Users/alvarem/Library/CloudStorage/GoogleDrive-', ...
    'miguelangel.alvarezballesteros@kaust.edu.sa/Other computers/', ...
    'My MacBook Pro/MEGA/0KAUST/0CSO_project/Figs'];

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
    exportgraphics(f, out_file, 'ContentType','vector');
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
    exportgraphics(f, out_file, 'ContentType','vector');
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
    exportgraphics(f, out_file, 'ContentType','vector');
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
    exportgraphics(f, out_file, 'ContentType','vector');
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





m_mix     = 1;
trace_mix = sa_pg_unbiased_mixture_clean( ...
    theta0, q0, r0, ...
    K_SA, ...
    Gamma, alpha, n0, ...
    m_mix, S_SA, ...
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
obs_pars_mix0.m_mix = m_mix;            % mixture parameter

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
    exportgraphics(f, out_file, 'ContentType','vector');
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
exportgraphics(f_tot, out_file_tot, 'ContentType','vector');

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
    exportgraphics(f, out_file, 'ContentType','vector');
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
exportgraphics(f_tot, out_file_tot, 'ContentType','vector');
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
    exportgraphics(f, out_file, 'ContentType','vector');
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
exportgraphics(f_tot, out_file_tot, 'ContentType','vector');
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
function [m_hist, P_hist] = kalman_1d(y, rho, q, r, H, m0, P0)
% y   : 1 × T
% rho : scalar state coefficient
% q   : process variance
% r   : observation variance
% H   : scalar observation matrix
% m0, P0 : prior mean/var for x_1
    T = size(y, 2);
    m = m0; P = P0;
    m_hist = zeros(1, T);
    P_hist = zeros(1, T);

    for t = 1:T
        % --- update with y_t ---
        S     = H*P*H' + r;              % innovation variance
        K     = (P*H') / S;              % Kalman gain
        innov = y(1, t) - H*m;           % innovation
        m     = m + K*innov;             % posterior mean
        P     = (1 - K*H)*P;             % posterior variance

        m_hist(t) = m;
        P_hist(t) = P;

        % --- predict to t+1 (skip after final step) ---
        if t < T
            m = rho*m;
            P = rho*P*rho + q;           % since scalar, rho' = rho
        end
    end
end

%%%
function out = pf_parallel( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    seed)
%pf_parallel_LOGW  Bootstrap PF (M i.i.d. filters) storing log-weights.
%
% out = pf_parallel_logw(y, T, N, M, in_dist_samp, in_pars, ...
%     trans_dist_samp, trans_pars, g, g_pars, seed)
%
% Inputs are identical to pf_parallel; this version additionally returns:
%   out.logw : N × T × M   (normalized log-weights at each time, pre-resampling)
%
% Notes:
% - Resamples EVERY step (independently per filter).
% - Uses log-sum-exp numerics; if all log-lik = -Inf for some filter/time,
%   sets weights to uniform so logw = -log(N).

if nargin < 11 || isempty(seed), seed = 12345; end

% --- basic checks ---
[~, T_chk] = size(y);
if T_chk ~= T
    error('size(y,2)=%d does not match T=%d.', T_chk, T);
end

% --- RNG: one stream per filter ---
streams = cell(1, M);
for m = 1:M
    streams{m} = RandStream('mrg32k3a', 'Seed', seed + m - 1);
end

% --- initialize particles at t=1 (pre-resampling) ---
X = in_dist_samp(in_pars, N, M);         % d_x × N × M
[d_x, N_chk, M_chk] = size(X);
if N_chk ~= N || M_chk ~= M
    error('in_dist_samp returned [%d %d %d], expected d_x×N×M with N=%d, M=%d.', ...
          d_x, N_chk, M_chk, N, M);
end

% --- preallocate outputs ---
particles = zeros(d_x, N, T, M);
ancestors = zeros(N, T, M, 'uint32');
logw_hist = -inf(N, T, M);   % normalized log-weights

% ===================== main loop =====================
for t = 1:T
    % store pre-resampling cloud
    particles(:, :, t, :) = X;

    % log-likelihoods: ll is 1×N×M
    ll = g(y(:, t), X, g_pars, t);
    sz = size(ll);
    if numel(sz) ~= 3 || sz(1) ~= 1 || sz(2) ~= N || sz(3) ~= M
        error('g must return 1×N×M; got [%s].', strjoin(string(sz), '×'));
    end

    % ---- normalize log-weights -> w_norm (N×M), with all-Inf guard ----
    L = squeeze(permute(ll, [2 3 1]));        % N × M  (log unnormalized weights)
    Lmax    = max(L, [], 1);
    shifted = L - Lmax;                        % log-sum-exp trick
    W       = exp(shifted);                    % N × M (unnormalized)
    S       = sum(W, 1);                       % 1 × M
    allInf  = (S == 0);

    % normalized weights
    w_norm = zeros(N, M);
    if any(~allInf), w_norm(:, ~allInf) = W(:, ~allInf) ./ S(1, ~allInf); end
    if any(allInf),  w_norm(:,  allInf) = 1 / N;                           end

    % ---- store normalized LOG-weights: logw = L - logsumexp(L) ----
    logw = -inf(N, M);
    if any(~allInf)
        denom = Lmax(1, ~allInf) + log(S(1, ~allInf));   % 1 × (#good columns)
        % subtract denom from each row in those columns
        logw(:, ~allInf) = bsxfun(@minus, L(:, ~allInf), denom);
    end
    if any(allInf)
        logw(:, allInf) = -log(N);   % uniform fallback → log(1/N)
    end
    % place into N × T × M at time t
    logw_hist(:, t, :) = reshape(logw, N, 1, M);

    % ---- multinomial resampling per filter (sorted-uniform scan) ----
    A = multinomial_resample_sorted(w_norm, streams);  % N×M (uint32)
    ancestors(:, t, :) = A;

    % ---- form post-resample cloud and propagate (except after final t) ----
    if t < T
        X_post = select_by_ancestors(X, A);            % d_x×N×M
        X = trans_dist_samp(X_post, trans_pars, t+1);  % d_x×N×M
        if ~isequal(size(X), [d_x, N, M])
            error('trans_dist_samp returned [%s], expected d_x×N×M.', ...
                  strjoin(string(size(X)), '×'));
        end
    end
end
% =====================================================

% package outputs
out.particles = particles;
out.ancestors = ancestors;
out.logw      = logw_hist;   % N × T × M

end % pf_parallel_logw

% ---------- helpers ----------

function A = multinomial_resample_sorted(w_norm, streams)
% w_norm: N×M (columns sum to 1)
% streams: {1×M} RandStream
    [N, M] = size(w_norm);
    A = zeros(N, M, 'uint32');
    for m = 1:M
        u   = sort(rand(streams{m}, N, 1));   % sorted uniforms
        cdf = cumsum(w_norm(:, m));           % cumulative weights
        i = 1; j = 1;
        while i <= N
            while u(i) > cdf(j)               % advance CDF pointer
                j = j + 1;
            end
            A(i, m) = uint32(j);
            i = i + 1;
        end
    end
end


function X_post = select_by_ancestors(X, A)
%SELECT_BY_ANCESTORS  Gather columns per filter using ancestor indices.
%   X_post = select_by_ancestors(X, A)
%   X : d_x × N × M       (particles, pre-resampling)
%   A : N × M (uint32)    (ancestor indices in 1..N per filter)
%   X_post : d_x × N × M  (post-resample cloud)
%
% Flip the mode below to 'vectorized' to use the no-loop gather.

    % ===== choose implementation here =====
    mode = "loop";    % "loop" or "vectorized"
    % ======================================

    % ---- basic checks ----
    if ndims(X) ~= 3
        error('X must be 3-D (d_x × N × M).');
    end
    if ~ismatrix(A)
        error('A must be 2-D (N × M).');
    end

    [d_x, N, M] = size(X);
    [Na, Ma]    = size(A);
    if Na ~= N || Ma ~= M
        error('Size mismatch: X is d_x×N×M = %d×%d×%d, but A is %d×%d.', d_x, N, M, Na, Ma);
    end
    if any(A(:) < 1 | A(:) > N)
        error('Ancestor indices in A must be within 1..N.');
    end

    switch mode
        case "loop"
            % Clear, parfor-ready per-filter gather (often fastest for small/medium M)
            X_post = zeros(d_x, N, M, 'like', X);
            for m = 1:M
                idx = double(A(:, m));        % 1..N indices for filter m
                X_post(:, :, m) = X(:, idx, m);
            end

        case "vectorized"
            % Vectorized gather using a single indexed take over stacked pages
            % 1) reshape pages side-by-side: d_x × (N*M)
            X23  = reshape(X, d_x, N*M);

            % 2) convert per-page indices to global 1..N*M via block offsets
            %    offs = [0, N, 2N, ..., (M-1)N]
            offs = (0:M-1) * N;                      % 1×M
            % Use bsxfun for broad MATLAB compatibility (instead of implicit expansion)
            J = bsxfun(@plus, double(A), offs);      % N×M

            % 3) gather, then reshape back to d_x × N × M
            X_post = reshape(X23(:, J(:)), d_x, N, M);

        otherwise
            error('Unknown mode "%s". Use "loop" or "vectorized".', mode);
    end
end
function out = cpf( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    x_ref, seed, ...
    traj_mode, trans_logpdf, ...
    store_particles, store_ancestors, store_logw)
%CPF_PARALLEL  Conditional bootstrap PF with selectable trajectory sampler.
%
% out = cpf_parallel(y, T, N, M, in_dist_samp, in_pars, trans_dist_samp, trans_pars, ...
%                    g, g_pars, x_ref, seed, traj_mode, trans_logpdf, ...
%                    store_particles, store_ancestors, store_logw)
%
% Required:
%   y                : d_y × T                  (shared across filters)
%   T, N, M          : time steps, #particles, #filters
%   in_dist_samp     : @(in_pars,N,M) -> d_x×N×M  initial particles
%   trans_dist_samp  : @(X_prev,trans_pars,t) -> d_x×N×M  transition sampler
%   g                : @(y_t,X_t,g_pars,t) -> 1×N×M       LOG-likelihoods
%   x_ref            : d_x × T × M             (reference paths, one per filter)
%
% Optional (defaults in code):
%   seed             : scalar RNG seed (default 12345)
%   traj_mode        : "ancestors" (default) or "backward"
%   trans_logpdf     : required iff traj_mode=="backward"
%                      @(x_next, X_prev, trans_pars, t) -> 1×N  LOG transition density
%   store_particles  : (logical) return out.particles (default true)
%   store_ancestors  : (logical) return out.ancestors (default true)
%   store_logw       : (logical) return out.logw      (default true)
%
% Outputs (always):
%   out.sampled_path : d_x × T × M    (sampled trajectory per filter)
%   out.sampled_idx  : 1 × M          (terminal index at time T)
%   out.sampled_mode : "ancestors" or "backward"
%
% Outputs (optional; see store_* flags):
%   out.particles    : d_x × N × T × M   (pre-resampling clouds)
%   out.ancestors    : N × T × M (uint32)
%   out.logw         : N × T × M   (normalized log-weights, pre-resampling)
%
% Notes:
% - Conditional path is fixed in slot j=1 across time (A(1,:)=1; X(:,1,t)=x_ref(:,t)).
% - Resampling is multinomial, independently per filter, every time step.
% - Internally, the function keeps the necessary arrays to produce the trajectory.
%   The store_* flags only control what is returned in 'out'.

% ----------------- defaults & input handling -----------------
if nargin < 12 || isempty(seed),       seed = 12345;        end
if nargin < 13 || isempty(traj_mode),  traj_mode = "ancestors"; end
traj_mode = lower(string(traj_mode));

if nargin < 14, trans_logpdf = []; end
if traj_mode == "backward" && ~isa(trans_logpdf,'function_handle')
    error('traj_mode="backward" requires a valid trans_logpdf function handle.');
end

if nargin < 15 || isempty(store_particles),  store_particles  = true; end
if nargin < 16 || isempty(store_ancestors),  store_ancestors  = true; end
if nargin < 17 || isempty(store_logw),       store_logw       = true; end

% ----------------- basic checks -----------------
[dy, T_chk] = size(y); %#ok<NASGU>
if T_chk ~= T, error('size(y,2)=%d does not match T=%d.', T_chk, T); end

% --- RNG: one stream per filter for clean independence ---
streams = cell(1, M);
for m = 1:M
    streams{m} = RandStream('mrg32k3a', 'Seed', seed + m - 1);
end

% --- initialize particles at t=1 (pre-resampling) ---
X = in_dist_samp(in_pars, N, M);           % d_x × N × M
[d_x, N_chk, M_chk] = size(X);
if N_chk ~= N || M_chk ~= M
    error('in_dist_samp returned [%d %d %d], expected d_x×N×M with N=%d, M=%d.', ...
          d_x, N_chk, M_chk, N, M);
end

% --- accept x_ref as d_x×T or d_x×T×1 and expand to d_x×T×M ---
if ndims(x_ref) == 2
    if ~isequal(size(x_ref), [d_x, T])
        error('x_ref must be d_x×T (or d_x×T×M). Got %s.', mat2str(size(x_ref)));
    end
    x_ref = reshape(x_ref, d_x, T, 1);
end
if size(x_ref,3) == 1 && M > 1
    x_ref = repmat(x_ref, 1, 1, M);
end
if ~isequal(size(x_ref), [d_x, T, M])
    error('x_ref must be d_x×T×M after expansion. Got %s.', mat2str(size(x_ref)));
end

% --- force reference path slot j=1 at t=1 ---
X(:,1,:) = x_ref(:,1,:);

% --- preallocate internal storage needed for trajectory construction ---
% We must keep particles & logw for all t to sample a trajectory.
particles_int = zeros(d_x, N, T, M, 'like', X);
ancestors_int = zeros(N, T, M, 'uint32');
logw_int      = -inf(N, T, M);

% ===================== main forward pass =====================
for t = 1:T
    % store pre-resampling cloud
    particles_int(:, :, t, :) = X;

    % log-likelihoods: 1×N×M
    ll = g(y(:, t), X, g_pars, t);
    if ~isequal(size(ll), [1 N M])
        error('g must return 1×N×M at t=%d; got %s.', t, mat2str(size(ll)));
    end

    % ---- normalize log-weights per filter (stable log-sum-exp) ----
    L = squeeze(permute(ll, [2 3 1]));      % N × M
    Lmax = max(L, [], 1);
    shifted = L - Lmax;
    W = exp(shifted);                        % N × M
    S = sum(W, 1);                           % 1 × M
    allInf = (S == 0);

    w_norm = zeros(N, M);
    if any(~allInf), w_norm(:, ~allInf) = W(:, ~allInf) ./ S(1, ~allInf); end
    if any(allInf),  w_norm(:,  allInf) = 1 / N;                           end

    % store normalized log-weights
    logw = -inf(N, M);
    if any(~allInf)
        denom = Lmax(1, ~allInf) + log(S(1, ~allInf));
        logw(:, ~allInf) = bsxfun(@minus, L(:, ~allInf), denom);
    end
    if any(allInf), logw(:, allInf) = -log(N); end
    logw_int(:, t, :) = reshape(logw, N, 1, M);

    % ---- multinomial resampling per filter; fix reference ancestor ----
    A = multinomial_resample_sorted(w_norm, streams);   % N×M
    A(1, :) = 1;                                        % keep ref in slot 1
    ancestors_int(:, t, :) = uint32(A);

    % ---- propagate to t+1 and re-impose reference in slot 1 ----
    if t < T
        X_post = select_by_ancestors(X, A);             % d_x×N×M
        X = trans_dist_samp(X_post, trans_pars, t+1);
        if ~isequal(size(X), [d_x, N, M])
            error('trans_dist_samp returned %s, expected d_x×N×M.', mat2str(size(X)));
        end
        X(:,1,:) = x_ref(:,t+1,:);                      % enforce conditional path at t+1
    end
end
% =============================================================

% ---- sample trajectory per filter (mode: "ancestors" or "backward") ----
sampled_idx  = zeros(1, M);
sampled_path = zeros(d_x, T, M, 'like', X);

% final weights
logw_T = reshape(logw_int(:, T, :), N, M);   % N×M (already log)
for m = 1:M
    % draw terminal index
    j = draw_from_logw_np(logw_T(:, m).', streams{m});
    sampled_idx(m) = j;

    if traj_mode == "ancestors"
        % backtrack via ancestors (note: use t-1)
        for t = T:-1:1
            sampled_path(:, t, m) = particles_int(:, j, t, m);
            if t > 1
                j = double(ancestors_int(j, t-1, m));
            end
        end

    elseif traj_mode == "backward"
        % backward simulation using trans_logpdf
        sampled_path(:, T, m) = particles_int(:, j, T, m);
        for t = T-1:-1:1
            x_next = sampled_path(:, t+1, m);              % d_x×1
            X_t    = particles_int(:, :, t, m);            % d_x×N
            lw_t   = (logw_int(:, t, m)).';                % 1×N
            lf     = trans_logpdf(x_next, X_t, trans_pars, t+1);   % 1×N
            if ~isequal(size(lf), [1, N])
                error('trans_logpdf must return 1×N at t=%d.', t+1);
            end
            logK = lw_t + lf;                               % 1×N
            j = draw_from_logw_np(logK, streams{m});
            sampled_path(:, t, m) = X_t(:, j);
        end
    else
        error('traj_mode must be "ancestors" or "backward".');
    end
end

% ----------------- package outputs (respect store_* flags) -----------------
out.sampled_path = sampled_path;
out.sampled_idx  = sampled_idx;
out.sampled_mode = char(traj_mode);

if store_particles, out.particles = particles_int; end
if store_ancestors, out.ancestors = ancestors_int; end
if store_logw,     out.logw      = logw_int;      end

end % cpf_parallel






function out = cpf_parallel( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    x_ref, seed, ...
    traj_mode, trans_logpdf, ...
    store_particles, store_ancestors, store_logw)
%CPF_PARALLEL  Conditional bootstrap PF with selectable trajectory sampler (parallelized per filter).
%
% [same header & notes as your version]

% ----------------- defaults & input handling -----------------
if nargin < 12 || isempty(seed),       seed = 12345;        end
if nargin < 13 || isempty(traj_mode),  traj_mode = "backward"; end
traj_mode = lower(string(traj_mode));

if nargin < 14, trans_logpdf = []; end
if traj_mode == "backward" && ~isa(trans_logpdf,'function_handle')
    error('traj_mode="backward" requires a valid trans_logpdf function handle.');
end

if nargin < 15 || isempty(store_particles),  store_particles  = true; end
if nargin < 16 || isempty(store_ancestors),  store_ancestors  = true; end
if nargin < 17 || isempty(store_logw),       store_logw       = true; end

% ----------------- basic checks -----------------
[~, T_chk] = size(y);
if T_chk ~= T, error('size(y,2)=%d does not match T=%d.', T_chk, T); end

% --- initialize particles at t=1 (pre-resampling) ---
X = in_dist_samp(in_pars, N, M);           % d_x × N × M
[d_x, N_chk, M_chk] = size(X);
if N_chk ~= N || M_chk ~= M
    error('in_dist_samp returned [%d %d %d], expected d_x×N×M with N=%d, M=%d.', ...
          d_x, N_chk, M_chk, N, M);
end

% --- accept x_ref as d_x×T or d_x×T×1 and expand to d_x×T×M ---
if ndims(x_ref) == 2
    if ~isequal(size(x_ref), [d_x, T])
        error('x_ref must be d_x×T (or d_x×T×M). Got %s.', mat2str(size(x_ref)));
    end
    x_ref = reshape(x_ref, d_x, T, 1);
end
if size(x_ref,3) == 1 && M > 1
    x_ref = repmat(x_ref, 1, 1, M);
end
if ~isequal(size(x_ref), [d_x, T, M])
    error('x_ref must be d_x×T×M after expansion. Got %s.', mat2str(size(x_ref)));
end

% --- force reference path slot j=1 at t=1 ---
X(:,1,:) = x_ref(:,1,:);

% --- preallocate internal storage needed for trajectory construction ---
particles_int = zeros(d_x, N, T, M, 'like', X);
ancestors_int = zeros(N, T, M, 'uint32');
logw_int      = -inf(N, T, M);

% ===================== main forward pass =====================
for t = 1:T
    % store pre-resampling cloud
    particles_int(:, :, t, :) = X;

    % allocate per-step containers (sliced by m for parfor)
    A_step    = zeros(N, M, 'uint32');
    logw_step = -inf(N, M);
    X_next    = zeros(d_x, N, M, 'like', X);   % only used if t<T

    % ===== BEGIN: parallel per-filter work =====
    parfor m = 1:M
        % deterministic RNG seed per (t,m)
        % (add a large stride per t so each time has an independent stream)
        seed_tm = seed + (m-1) + 100000*(t-1);
        try
            rng(seed_tm, 'Threefry'); %#ok<RNGB>
        catch
            rng(seed_tm, 'twister');  %#ok<RNGB> % fallback if Threefry not available
        end

        % ---- log-likelihoods for this filter (shape it to N×1) ----
        ll_m = g(y(:, t), X(:, :, m), g_pars, t);   % expected 1×N or 1×N×1
        L = ll_m(:);                                % N×1

        % ---- normalize (log-sum-exp) ----
        Lmax = max(L);
        W = exp(L - Lmax);
        S = sum(W);
        if S == 0 || ~isfinite(S)
            w = ones(N,1) / N;
            logw_m = (-log(N)) * ones(N,1);
        else
            w = W / S;
            logw_m = L - (Lmax + log(S));
        end
        logw_step(:, m) = logw_m;

        % ---- multinomial resampling (sorted uniforms) ----
        u = sort(rand(N,1));
        c = cumsum(w);
        Ai = zeros(N,1,'uint32');
        i = 1; j = 1;
        while i <= N
            while u(i) > c(j), j = j + 1; end
            Ai(i) = uint32(j); i = i + 1;
        end
        Ai(1) = uint32(1);   % keep reference in slot 1
        A_step(:, m) = Ai;

        % ---- propagate to t+1 (form post-resample cloud locally) ----
        if t < T
            X_post_m = X(:, double(Ai), m);                  % d_x×N
            Xn = trans_dist_samp(X_post_m, trans_pars, t+1); % d_x×N
            Xn(:,1) = x_ref(:, t+1, m);                      % enforce conditional path
            X_next(:, :, m) = Xn;
        end
    end
    % ===== END: parallel per-filter work =====

    % collect this step's results
    logw_int(:, t, :)      = reshape(logw_step, N, 1, M);
    ancestors_int(:, t, :) = A_step;

    if t < T
        X = X_next;  % move to next time
    end
end
% =============================================================

% ---- sample trajectory per filter (mode: "ancestors" or "backward") ----
sampled_idx  = zeros(1, M);
sampled_path = zeros(d_x, T, M, 'like', X);

% final weights
wT = exp(logw_int(:, T, :));    % N × 1 × M
wT = reshape(wT, N, M);         % N × M

% ===== BEGIN: parallel backtrace per filter =====
parfor m = 1:M
    % local RNG for terminal draw (keep deterministic per m)
    seed_Tm = seed + (m-1) + 100000*T;
    try
        rng(seed_Tm, 'Threefry'); %#ok<RNGB>
    catch
        rng(seed_Tm, 'twister');  %#ok<RNGB>
    end

    % draw terminal index from final weights
    jT = draw_from_logw(log(wT(:, m).'+realmin));  % takes 1×N logw row
    sampled_idx(m) = jT;

    % build the whole slice locally, then assign once
    sp = zeros(d_x, T, 'like', X);   % local buffer for sampled_path(:,:,m)

    if traj_mode == "ancestors"
        j = jT;
        for t = T:-1:1
            sp(:, t) = particles_int(:, j, t, m);
            %sampled_path(:, t, m) = particles_int(:, j, t, m);
            if t > 1
                j = double(ancestors_int(j, t-1, m));
            end
        end

    elseif traj_mode == "backward"
        j = jT;
        sp(:, T) = particles_int(:, j, T, m);
        %sampled_path(:, T, m) = particles_int(:, j, T, m);
        for t = T-1:-1:1
            x_next = sp(:, t+1);         % d_x×1
            %x_next = sampled_path(:, t+1, m);         % d_x×1
            X_t    = particles_int(:, :, t, m);       % d_x×N
            lw_t   = (logw_int(:, t, m)).';           % 1×N
            lf     = trans_logpdf(x_next, X_t, trans_pars, t+1); % 1×N
            if ~isequal(size(lf), [1, N])
                error('trans_logpdf must return 1×N at t=%d.', t+1);
            end
            logK = lw_t + lf;                          % 1×N
            j = draw_from_logw(logK);
            sp(:, t) = X_t(:, j);
            %sampled_path(:, t, m) = X_t(:, j);
        end
    else
        error('traj_mode must be "ancestors" or "backward".');
    end
    sampled_path(:, :, m) = sp;
end
% ===== END: parallel backtrace per filter =====

% ----------------- package outputs (respect store_* flags) -----------------
out.sampled_path = sampled_path;
out.sampled_idx  = sampled_idx;
out.sampled_mode = char(traj_mode);

if store_particles, out.particles = particles_int; end
if store_ancestors, out.ancestors = ancestors_int; end
if store_logw,     out.logw      = logw_int;      end

end % cpf_parallel


% ----------------------- helpers -----------------------
function j = draw_from_logw(logw_row)
% Draw index (1..N) from a 1×N vector of (un-normalized) log-weights.
    if size(logw_row,1) ~= 1, logw_row = logw_row(:).'; end
    Lmax = max(logw_row);
    x = exp(logw_row - Lmax);
    s = sum(x);
    if s == 0 || ~isfinite(s)
        % fallback: uniform
        N = numel(logw_row);
        j = 1 + floor(rand(1) * N);
        return;
    end
    p = x / s;
    u = rand(1);
    c = cumsum(p);
    j = find(u <= c, 1, 'first');
end


function j = draw_from_logw_np(logw_row, stream)
% Draw index (1..N) from a 1×N vector of log-weights (unnormalized is OK).
% Stable: subtract max, exp, normalize, CDF, one uniform.
    if size(logw_row,1) ~= 1
        logw_row = logw_row(:).';
    end
    Lmax = max(logw_row);
    x = exp(logw_row - Lmax);
    s = sum(x);
    if s == 0 || ~isfinite(s)
        % fallback: uniform
        j = 1 + floor(rand(stream,1) * numel(logw_row));
        return;
    end
    p = x / s;
    u = rand(stream,1);
    c = cumsum(p);
    j = find(u <= c, 1, 'first');
end

% ----------------------- helpers -----------------------

%% ======================= Local function: RTS smoother =======================
function [m_f, P_f, m_s, P_s] = rts_smoother_1d(y, rho, q, r, H, m0, P0)
% Forward: Kalman filter (update at t, then predict to t+1)
T = size(y,2);
m_f = zeros(1,T); P_f = zeros(1,T);
m = m0; P = P0;
for t = 1:T
    % Update with y_t
    S = H*P*H' + r;                 % innovation variance
    K = (P*H') / S;                 % gain
    innov = y(1,t) - H*m;
    m = m + K*innov;
    P = (1 - K*H)*P;
    m_f(t) = m; P_f(t) = P;
    % Predict to t+1
    if t < T
        m = rho*m;
        P = rho*P*rho + q;
    end
end

% Backward: RTS smoothing
m_s = zeros(1,T); P_s = zeros(1,T);
m_s(T) = m_f(T); P_s(T) = P_f(T);
for t = T-1:-1:1
    % Predict stats from t to t+1 (using filtered at t)
    m_pred = rho * m_f(t);
    P_pred = rho * P_f(t) * rho + q;
    % Smoother gain
    C = (P_f(t) * rho) / P_pred;
    % Smoothed mean/var
    m_s(t) = m_f(t) + C * (m_s(t+1) - m_pred);
    P_s(t) = P_f(t) + C^2 * (P_s(t+1) - P_pred);
end
end

function out = pgibbs_run_init_pfmean( ...
    y, T, N, M, B, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    seed0, ...
    cpf_choice, ...
    traj_mode, trans_logpdf, ...
    store_particles, store_ancestors, store_logw, ...
    N_init, init_mode)

% Defaults
if nargin < 13 || isempty(cpf_choice),   cpf_choice   = "cpf_parallel"; end
if nargin < 14 || isempty(traj_mode),    traj_mode    = "ancestors";    end
if nargin < 15, trans_logpdf = []; end
if nargin < 16 || isempty(store_particles), store_particles = false; end
if nargin < 17 || isempty(store_ancestors), store_ancestors = false; end
if nargin < 18 || isempty(store_logw),      store_logw      = false; end
if nargin < 19 || isempty(N_init),          N_init          = N;     end
if nargin < 20 || isempty(init_mode),       init_mode       = "weighted"; end

cpf_choice = lower(string(cpf_choice));
traj_mode  = lower(string(traj_mode));
init_mode  = lower(string(init_mode));

if ~(cpf_choice=="cpf" || cpf_choice=="cpf_parallel")
    error('cpf_choice must be "cpf" or "cpf_parallel".');
end
if traj_mode=="backward" && ~isa(trans_logpdf,'function_handle')
    error('traj_mode="backward" requires trans_logpdf.');
end
if size(y,2) ~= T, error('size(y,2)=%d ≠ T=%d', size(y,2), T); end

% Probe d_x
X0 = in_dist_samp(in_pars, max(1,N), max(1,M));
d_x = size(X0,1);

% ---- 1) Build M independent initial paths from PF(s) ----
x_init_paths = pf_weighted_mean_paths( ...
    y, T, N_init, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, seed0, init_mode);       % d_x×T×M

% ---- 2) PG over B links, store B+1 paths ----
X_paths = zeros(d_x, T, M, B+1, 'like', x_init_paths);
X_paths(:,:,:,1) = x_init_paths;

cpf_fun = str2func(char(cpf_choice));
time_per_iter = zeros(1, B);

x_ref = x_init_paths;
for b = 1:B
    t0 = tic;
    out_cpf = cpf_fun(y, T, N, M, ...
                      in_dist_samp, in_pars, ...
                      trans_dist_samp, trans_pars, ...
                      g, g_pars, ...
                      x_ref, seed0 + (b-1), ...
                      traj_mode, trans_logpdf, ...
                      store_particles, store_ancestors, store_logw);
    time_per_iter(b) = toc(t0);

    X_paths(:,:,:,b+1) = out_cpf.sampled_path;
    x_ref = out_cpf.sampled_path;   % condition next step on current sample
end

% ---- 3) package ----
out.X_paths       = X_paths;           % d_x×T×M×(B+1)
out.time_per_iter = time_per_iter;
out.init_from     = sprintf('PF %s mean (N_init=%d, M=%d)', char(init_mode), N_init, M);
out.settings = struct('T',T,'N',N,'M',M,'B',B, ...
                      'cpf_choice',char(cpf_choice), ...
                      'traj_mode',char(traj_mode), ...
                      'seed0',seed0, ...
                      'N_init',N_init, 'init_mode',char(init_mode), ...
                      'store_particles',store_particles, ...
                      'store_ancestors',store_ancestors, ...
                      'store_logw',store_logw);
end


% ===== helper: run a PF and return the weighted filtered mean path (d_x×T) =====
function x_init_paths = pf_weighted_mean_paths( ...
    y, T, N_init, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, seed, init_mode)

% init_mode: "weighted" (default) or "resampled"

if nargin < 13 || isempty(init_mode), init_mode = "weighted"; end
init_mode = lower(string(init_mode));

% (Optional) make g robust to M=1 by forcing 1×N×M shape
g_fix = @(yt,Xt,p,t) reshape( ...
           g(yt,Xt,p,t), 1, size(Xt,2), size(Xt,3));

% Run one PF with M filters; we need particles & logw
out_pf = pf_parallel(y, T, N_init, M, ...
                     in_dist_samp, in_pars, ...
                     trans_dist_samp, trans_pars, ...
                     g_fix, g_pars, seed);

if ~isfield(out_pf,'particles') || ~isfield(out_pf,'logw')
    error('pf_parallel must return .particles (d_x×N×T×M) and .logw (N×T×M).');
end

X  = out_pf.particles;   % d_x × N × T × M (pre-resampling clouds)
lw = out_pf.logw;        % N × T × M (normalized log-weights, log)
w  = exp(lw);            % N × T × M

[d_x, N, T, M] = size(X);

switch init_mode
case "weighted"
    % x_init(:,t,m) = sum_i w(i,t,m) * X(:,i,t,m)

    tmp = sum( X .* reshape(w, 1, N, T, M), 2);  % d_x × 1 × T × M
    x_init_paths = reshape(tmp, size(X,1), T, M);  % d_x × T × M

case "resampled"
    % Build a resampled cloud per (t,m), then average (adds variance but uses ancestors idea)
    x_init_paths = zeros(d_x, T, M, 'like', X);
    for m = 1:M
        for t = 1:T
            % multinomial draw N indices from weights w(:,t,m)
            pm = w(:,t,m) ./ max(sum(w(:,t,m)), realmin);
            u  = sort(rand(N,1));
            c  = cumsum(pm);
            idx = zeros(N,1,'uint32');
            i=1; j=1;
            while i<=N
                while u(i)>c(j), j=j+1; end
                idx(i)=uint32(j); i=i+1;
            end
            Xrt = X(:, double(idx), t, m);      % resampled cloud
            x_init_paths(:, t, m) = mean(Xrt, 2);
        end
    end

otherwise
    error('init_mode must be "weighted" or "resampled".');
end
end

function [score, parts] = score_gaussian_ssm(y, theta, q, r, S0)
% SCORE_GAUSSIAN_SSM  Gradient (score) of the log-likelihood for a 1D LGSSM.
% Model:
%   X1 ~ N(0, S0)
%   Xt | X_{t-1} ~ N(theta * X_{t-1}, q)
%   Yt | Xt ~ N(Xt, r)
%
% Inputs
%   y      : 1×T observations
%   theta  : scalar (state coefficient)
%   q      : scalar > 0 (state var)
%   r      : scalar > 0 (obs var)
%   S0     : scalar > 0 (initial var)
%
% Outputs
%   score  : struct with fields dtheta, dq, dr, dS0
%   parts  : struct with useful internals (filtered, smoothed, cross-cov, etc.)

    y  = y(:)';                 % 1×T
    T  = size(y,2);
    eps_small = 1e-12;

    % ---------- Forward: Kalman filter (update at t, then predict to t+1)
    m_f = zeros(1,T);  P_f = zeros(1,T);
    m_pred = zeros(1,T);  P_pred = zeros(1,T);   % store P_{t|t-1} (shifted)
    % prior for t=1 (before seeing y1)
    m = 0;
    P = S0;
    for t = 1:T
        % innovation update at t
        S_t = P + r;                          % 1×1
        K   = P / S_t;
        innov = y(t) - m;
        m = m + K * innov;
        P = (1 - K) * P;
        m_f(t) = m;  P_f(t) = P;

        % prediction to t+1
        if t < T
            m_pred(t+1) = theta * m;
            P_pred(t+1) = theta^2 * P + q;
            m = m_pred(t+1);
            P = P_pred(t+1);
        end
    end

    % ---------- Backward: RTS smoother + lag-one covariance
    m_s = zeros(1,T);  P_s = zeros(1,T);
    J   = zeros(1,T-1);           % smoother gains J_t for t=1..T-1
    C_lag = zeros(1,T);           % Cov(X_t, X_{t-1} | y), defined for t>=2

    m_s(T) = m_f(T);  P_s(T) = P_f(T);
    for t = T-1:-1:1
        % P_pred(t+1) is prediction variance from t to t+1
        Pp = max(P_pred(t+1), eps_small);
        J(t) = (P_f(t) * theta) / Pp;

        % smooth
        m_s(t) = m_f(t) + J(t) * (m_s(t+1) - theta*m_f(t));
        P_s(t) = P_f(t) + J(t)^2 * (P_s(t+1) - Pp);

        % lag-one covariance: Cov(X_t, X_{t-1} | y), here for index (t) vs (t-1)
        % Use Σ_{t-1,t}^s = J(t-1) * Σ_t^s  ⇒ Cov(X_t, X_{t-1}) = J(t-1) * P_s(t)
        % We'll fill it after loop for t>=2 using J(t-1)
    end
    for t = 2:T
        C_lag(t) = J(t-1) * P_s(t);   % Cov(X_t, X_{t-1} | y)
    end

    % ---------- Expectations needed for the Fisher score
    % E[X_t] = m_s(t);  Var[X_t] = P_s(t);
    % E[X_t^2] = P_s(t) + m_s(t)^2
    EX2  = P_s + m_s.^2;
    % E[X_t X_{t-1}] = Cov + mean product
    EXXt = C_lag + m_s .* [0, m_s(1:end-1)];   % first entry unused (t=1)

    % ---------- Score components
    % (1) wrt theta:  sum_{t=2..T} (1/q) E[(X_t - theta X_{t-1}) X_{t-1}]
    %     = (1/q) sum_{t=2..T} (E[X_t X_{t-1}] - theta E[X_{t-1}^2])
    E_XtXm1   = EXXt(2:end);
    E_Xm1sq   = EX2(1:end-1);
    dtheta = (1/max(q,eps_small)) * sum( E_XtXm1 - theta * E_Xm1sq );

    % (2) wrt q:  -(T-1)/(2q) + (1/(2q^2)) sum_{t=2..T} E[(X_t - theta X_{t-1})^2]
    % E[(X_t - theta X_{t-1})^2] = E[X_t^2] - 2theta E[X_t X_{t-1}] + theta^2 E[X_{t-1}^2]
    E_res2 = EX2(2:end) - 2*theta*E_XtXm1 + theta^2 * E_Xm1sq;
    dq = -(T-1)/(2*max(q,eps_small)) + 0.5 * sum(E_res2) / max(q,eps_small)^2;

    % (3) wrt r:  -T/(2r) + (1/(2r^2)) sum_t E[(Y_t - X_t)^2]
    % E[(Y_t - X_t)^2] = (y - m_s).^2 + P_s
    E_meas2 = (y - m_s).^2 + P_s;
    dr = -T/(2*max(r,eps_small)) + 0.5 * sum(E_meas2) / max(r,eps_small)^2;

    % (4) wrt S0 (initial variance):  -1/(2S0) + (1/(2S0^2)) E[X_1^2]
    dS0 = -1/(2*max(S0,eps_small)) + 0.5 * EX2(1) / max(S0,eps_small)^2;

    % ---------- Package
    score = struct('dtheta', dtheta, 'dq', dq, 'dr', dr, 'dS0', dS0);

    if nargout > 1
        parts = struct();
        parts.m_f = m_f; parts.P_f = P_f;
        parts.m_s = m_s; parts.P_s = P_s;
        parts.J = J; parts.P_pred = P_pred;
        parts.C_lag = C_lag;            % Cov(X_t, X_{t-1} | y), t>=2
        parts.EX2 = EX2; parts.EXXt = EXXt;
        parts.E_res2 = E_res2; parts.E_meas2 = E_meas2;
    end
end
function g_obs = grad_log_g_t_stud_vec(y, X_paths, obs_pars)
%GRAD_LOG_G_T_STUD_VEC  Vectorized obs-score wrt log(sigma)
%
% Inputs:
%   y        : 1×T
%   X_paths  : 1×T×P
%   obs_pars : struct with .v, .sigma
%
% Output:
%   g_obs : 1×P  (sum over t of gradient wrt log(sigma))

    v     = obs_pars.v;
    sigma = obs_pars.sigma;

    % y:      1×T        → 1×T×1 for broadcast
    % X_paths:1×T×P
    diff = y - X_paths;               % 1×T×P
    z    = diff.^2;                   % 1×T×P

    denom = v*sigma^2 + z;            % 1×T×P

    dlog_sigma_tp = -1 + (v+1) .* (z ./ denom);   % 1×T×P

    % sum over time dimension → 1×1×P → reshape to 1×P
    g_obs = squeeze(sum(dlog_sigma_tp, 2));       % 1×P
end


function dlog_sigma = grad_log_g_t_stud(yt, Xt, obs_pars, ~)
%GRAD_LOG_G_T  Gradient of t-loglik wrt log(sigma), per particle.
%
%   dlog_sigma = grad_log_g_t(yt, Xt, obs_pars, t)
%
% Inputs:
%   yt       : 1×1 or 1×M
%   Xt       : 1×N×M
%   obs_pars : struct with fields
%                .v     : degrees of freedom ν
%                .sigma : scale σ
%
% Output:
%   dlog_sigma : 1×N×M  gradient wrt log(sigma) for each particle

    v     = obs_pars.v;
    sigma = obs_pars.sigma;

    diff = yt - Xt;            % 1×N×M
    z    = diff.^2;            % 1×N×M

    % gradient wrt log(sigma)
    dlog_sigma = -1 + (v+1) .* (z ./ (v*sigma^2 + z));  % 1×N×M
end



% function for the gradients of the logs of the ssm.
function g = grad_log_g_gauss(y_t, x_t, obs_pars, ~)
% y_t, x_t are scalars here (1×1), obs_pars.R = r
r = obs_pars.R;
res2 = (y_t - x_t).^2;
g = -0.5/r + 0.5*res2/(r^2);   % scalar gradient wrt r
end

function ll = log_g_t_stud(yt, Xt, obs_pars, ~)
%LOG_G_T_STUD  Log-likelihood for Student-t observation model
%
%   ll = log_g_t_stud(yt, Xt, obs_pars, t)
%
% Inputs:
%   yt       : 1×1 or 1×M (1D observation at time t)
%   Xt       : 1×N×M     (particles x_t for each filter)
%   obs_pars : struct with fields
%                .v     : degrees of freedom ν
%                .sigma : scale σ > 0
%
% Output:
%   ll : 1×N×M  log p(yt | Xt, v, sigma)

    v     = obs_pars.v;
    sigma = obs_pars.sigma;

    % residual
    diff = yt - Xt;        % 1×N×M
    z    = diff.^2;        % 1×N×M

    const = gammaln((v+1)/2) - gammaln(v/2) ...
          - 0.5*log(v*pi) - log(sigma);

    ll = const ...
       - 0.5*(v+1) .* log(1 + z ./ (v*sigma^2));   % 1×N×M
end


function g = grad_log_f_gauss(x_t, x_tm1, trans_pars, ~)
% trans_pars: .theta, .q
theta = trans_pars.theta; q = trans_pars.q;
res   = x_t - theta*x_tm1;
dtheta = (res * x_tm1) / q;
dq     = -0.5/q + 0.5*(res.^2)/(q^2);
g = [dtheta; dq];               % 2×1
end

function g = grad_log_p1_gauss(x1, init_pars)
% init_pars.S0
S0 = init_pars.S0;
g  = -0.5/S0 + 0.5*(x1.^2)/(S0^2);  % scalar
end

function out = score_from_paths( ...
    y, X_paths, ...
    grad_log_g, grad_log_f, grad_log_p1, ...
    obs_pars, trans_pars, init_pars, ...
    paths_sel, weights, mode)

% SCORE_FROM_PATHS  Score (gradient) from PG-sampled paths.
% Supports:
%   mode="sequential"         (generic; uses callbacks as-is)
%   mode="vectorized_gauss1d" (fast path for 1D Gaussian AR(1): r, theta, q, S0)
%   mode="vectorized_callbacks"
% Inputs
%   y            : d_y × T
%   X_paths      : d_x × T × M × B1
%   grad_log_*   : callbacks (only used in "sequential" mode)
%   *_pars       : structs with parameters (for gauss1d: obs_pars.R=r; trans_pars.theta,q; init_pars.S0)
%   paths_sel    : logical M×B1 (optional; default all true)
%   weights      : P×1 (optional; default uniform)
%   mode         : "sequential" (default) | "vectorized_gauss1d"
%
% Output
%   out.avg_obs   : gradient wrt observation parameters
%   out.avg_trans : gradient wrt transition  parameters
%   out.avg_init  : gradient wrt initial     parameters
%   out.avg_total : concatenation [obs; trans; init]

    if nargin < 9 || isempty(paths_sel)
        paths_sel = true(size(X_paths,3), size(X_paths,4));
    end
    if nargin < 10 || isempty(weights)
        % filled after we know P
    end
    if nargin < 11 || isempty(mode)
        mode = "sequential";
    else
        mode = string(mode);
    end

    [d_x, T, M, B1] = size(X_paths); %#ok<NASGU>
    idx = find(paths_sel(:));
    P   = numel(idx);
    if P == 0
        error('paths_sel selects zero paths.');
    end

    % Extract selected paths to d_x × T × P
    Xp = zeros(size(X_paths,1), size(X_paths,2), P, 'like', X_paths);
    [mm, bb] = ind2sub([size(X_paths,3), size(X_paths,4)], idx);
    for k = 1:P
        Xp(:,:,k) = X_paths(:,:,mm(k), bb(k));
    end

    % weights
    if nargin < 10 || isempty(weights)
        weights = ones(P,1, 'like', Xp) / P;
    else
        weights = weights(:) / sum(weights);
    end

    switch lower(mode)
        case "sequential"
            % ----------------- generic, model-agnostic -----------------
            g_obs   = 0;    % size set by first callback return
            g_trans = 0;
            g_init  = 0;

            for k = 1:P
                x = Xp(:,:,k);             % d_x × T
                g0 = grad_log_p1(x(:,1), init_pars);
                gt = 0; go = 0;
                for t = 1:T
                    go = go + grad_log_g(y(:,t), x(:,t), obs_pars, t);
                    if t >= 2
                        gt = gt + grad_log_f(x(:,t), x(:,t-1), trans_pars, t);
                    end
                end
                w = weights(k);
                g_init  = g_init  + w * g0;
                g_trans = g_trans + w * gt;
                g_obs   = g_obs   + w * go;
            end

        case "vectorized_callbacks"
    % ================= vectorized via user callbacks =================
    % Expect the callbacks to accept all paths at once and return
    % per-time, per-path contributions to be reduced here.
    %
    % Required vectorized signatures:
    %   Gg = grad_log_g( y, Xp, obs_pars )
    %       y   : d_y × T                  (shared)
    %       Xp  : d_x × T × P              (all selected paths)
    %       Gg  : p_obs × T × P            (per-time, per-path contributions)
    %
    %   Gf = grad_log_f( Xp, trans_pars )
    %       Xp  : d_x × T × P
    %       Gf  : p_tr  × (T-1) × P        (per-time (t=2..T), per-path)
    %
    %   G0 = grad_log_p1( X1, init_pars )
    %       X1  : d_x × 1 × P              (just the initial state per path)
    %       G0  : p_init × P                (per-path)
    %
    % Notes:
    % - We sum across time inside this branch, then average across paths
    %   using 'weights' (P×1).
    % - Shapes p_obs, p_tr, p_init can be any positive integers.

    % 1) Observation term: p_obs×T×P  -> sum over T -> p_obs×P
    Gg = grad_log_g(y, Xp, obs_pars);                 % p_obs×T×P
    if ndims(Gg) ~= 3 || size(Gg,2) ~= T || size(Gg,3) ~= P
        error('grad_log_g must return p_obs×T×P; got %s.', mat2str(size(Gg)));
    end
    Gg_sum = squeeze(sum(Gg, 2));                     % p_obs×P

    % 2) Transition term: p_tr×(T-1)×P -> sum over (T-1) -> p_tr×P
    Gf = grad_log_f(Xp, trans_pars);                  % p_tr×(T-1)×P
    if ndims(Gf) ~= 3 || size(Gf,2) ~= (T-1) || size(Gf,3) ~= P
        error('grad_log_f must return p_tr×(T-1)×P; got %s.', mat2str(size(Gf)));
    end
    Gf_sum = squeeze(sum(Gf, 2));                     % p_tr×P

    % 3) Initial term: p_init×P (already per-path)
    X1 = Xp(:,1,:);                                   % d_x×1×P
    G0 = grad_log_p1(X1, init_pars);                  % p_init×P
    if ~ismatrix(G0) || size(G0,2) ~= P
        error('grad_log_p1 must return p_init×P; got %s.', mat2str(size(G0)));
    end

    % 4) Weighted averages across paths (broadcast weights: 1×P)
    wrow = reshape(weights, 1, P);                    % 1×P

    % observation gradient: p_obs×P  • w -> p_obs×1
    g_obs   = Gg_sum * weights;                       % p_obs×1

    % transition gradient: p_tr×P    • w -> p_tr×1
    g_trans = Gf_sum * weights;                       % p_tr×1

    % initial gradient:    p_init×P  • w -> p_init×1
    g_init  = G0     * weights;                       % p_init×1


        case "vectorized_gauss1d"
            % ----------------- fast path: 1D Gaussian AR(1) -----------------
            % Checks
            if size(Xp,1) ~= 1
                error('vectorized_gauss1d requires d_x = 1.');
            end
            if ~isfield(obs_pars,'R') || ~isfield(trans_pars,'theta') || ~isfield(trans_pars,'q') || ~isfield(init_pars,'S0')
                error('vectorized_gauss1d expects fields: obs_pars.R, trans_pars.theta,q, init_pars.S0.');
            end

            r     = obs_pars.R;
            theta = trans_pars.theta; q = trans_pars.q;
            S0    = init_pars.S0;

            % shapes: squeeze to T×P (since d_x=1)
            Xall = squeeze(Xp);           % T × P
            if isrow(Xall), Xall = Xall.'; end
            yrow = y(:).';                % 1 × T

            % --- observation term (dr)
            diff_yx = yrow.' - Xall;      % T × P
            dr_each = -0.5./r + 0.5*(diff_yx.^2)./(r.^2);   % T × P
            dr_k    = sum(dr_each, 1);                        % 1 × P
            g_obs   = sum(weights.' .* dr_k);                % scalar

            % --- transition term (dtheta, dq)
            x_t   = Xall(2:end, :);        % (T-1) × P
            x_tm1 = Xall(1:end-1, :);      % (T-1) × P
            res   = x_t - theta .* x_tm1;  % (T-1) × P

            dtheta_each = (res .* x_tm1) ./ q;               % (T-1) × P
            dq_each     = -0.5./q + 0.5*(res.^2)./(q.^2);    % (T-1) × P

            dtheta_k = sum(dtheta_each, 1);                  % 1 × P
            dq_k     = sum(dq_each, 1);                      % 1 × P

            dtheta = sum(weights.' .* dtheta_k);             % scalar
            dq     = sum(weights.' .* dq_k);                 % scalar
            g_trans = [dtheta; dq];

            % --- initial term (dS0)
            x1     = Xall(1, :);                             % 1 × P
            dS0_k  = -0.5./S0 + 0.5*(x1.^2)./(S0.^2);        % 1 × P
            g_init = sum(weights.' .* dS0_k);                % scalar

        otherwise
            error('Unknown mode "%s". Use "sequential" or "vectorized_gauss1d".', mode);
    end

    out = struct();
    out.avg_obs   = g_obs;
    out.avg_trans = g_trans;
    out.avg_init  = g_init;
    % concatenate in the natural order used elsewhere: [dr; dtheta; dq; dS0] for gauss1d
    try
        out.avg_total = [g_obs; g_trans; g_init];
    catch
        % If shapes don't concatenate (e.g., model-specific vectors), skip total
        out.avg_total = [];
    end
end
%%
function out = score_gaussian_from_paths(y, X_paths, theta, q, r, S0, exclude_first_col)
% y: 1×T, X_paths: 1×T×M×B1
if nargin < 7, exclude_first_col = false; end
obs_pars   = struct('R', r);
trans_pars = struct('theta', theta, 'q', q);
init_pars  = struct('S0', S0);

[M, B1] = deal(size(X_paths,3), size(X_paths,4));
paths_sel = true(M,B1);
if exclude_first_col, paths_sel(:,1) = false; end

out = score_from_paths(y, X_paths, ...
        @grad_log_g_gauss, @grad_log_f_gauss, @grad_log_p1_gauss, ...
        obs_pars, trans_pars, init_pars, paths_sel, []);
end


%% ===== helper: one run =====
function [enorm, ecomp, tsec] = one_run_score_err( ...
    y,T,N,M,B, ...
    in_dist,in_pars,trans,tr_pars,g,g_pars, ...
    seed0, ...
    cpf_choice,traj_mode,trans_logpdf, ...
    store_parts,store_anc,store_logw, ...
    N_init,init_mode, g_anal)

t0 = tic;
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
tsec = toc(t0);

X_paths = out_pg.X_paths;                  % 1×T×M×(B+1)
S_mc    = score_gaussian_from_paths_vectorized(y, X_paths, tr_pars.theta, tr_pars.q, g_pars.R, in_pars.Sigma, true);
g_mc    = [S_mc.avg_obs; S_mc.avg_trans; S_mc.avg_init];  % [dr; dtheta; dq; dS0]
diff    = g_mc - g_anal;
enorm   = norm(diff,2);
ecomp   = abs(diff).';
end

%%
% Observation gradient (wrt r): returns p_obs×T×P with p_obs=1
function g_obs = grad_log_g_gauss_vec(y, X_paths, obs_pars)
%GRAD_LOG_G_GAUSS_VEC  Vectorized obs-score wrt log(sigma) for Gaussian obs
%
% Model:
%   Y_t | X_t ~ N(X_t, sigma^2),  with sigma^2 = R (or given explicitly).
%
% Inputs:
%   y        : 1×T
%   X_paths  : 1×T×P          (P trajectories)
%   obs_pars : struct with either
%                .R      : variance  (then sigma^2 = R)
%              or
%                .sigma  : std dev   (then variance = sigma^2)
%
% Output:
%   g_obs : 1×P  (sum over t of d/d(log sigma) log p(y_t | x_t))
%
% Formula for a single (t, path) entry:
%   d/d(log sigma) log N(y_t; x_t, sigma^2)
%      = -1 + (y_t - x_t)^2 / sigma^2

    % --- get sigma^2 consistently ---
    if isfield(obs_pars, 'sigma')
        sigma = obs_pars.sigma;
        sig2  = sigma.^2;
    elseif isfield(obs_pars, 'R')
        sig2  = obs_pars.R;
        sigma = sqrt(sig2);   %#ok<NASGU>  % only for completeness
    else
        error('obs_pars must contain either .R or .sigma');
    end

    % y:      1×T       → broadcast to 1×T×P
    % X_paths:1×T×P
    diff = y - X_paths;          % 1×T×P
    z    = diff.^2;              % 1×T×P

    % per (t,p) derivative wrt log sigma
    dlog_sigma_tp = -1 + z ./ sig2;   % 1×T×P

    % sum over time dimension → 1×1×P → reshape to 1×P
    g_obs = squeeze(sum(dlog_sigma_tp, 2));  % 1×P
end
% Transition gradient (wrt [theta; q]): returns p_tr×(T-1)×P with p_tr=2
function Gf = grad_log_f_gauss_vec(Xp, trans_pars)
% Xp : 1×T×P
% Gf : 2×(T-1)×P  (row 1: dtheta, row 2: dq)
theta = trans_pars.theta;
q     = trans_pars.q;
x_tm1 = Xp(:, 1:end-1, :);                   % 1×(T-1)×P
x_t   = Xp(:, 2:end,   :);                   % 1×(T-1)×P
res   = x_t - theta .* x_tm1;                % 1×(T-1)×P

dtheta = (res .* x_tm1) ./ q;                % 1×(T-1)×P
dq     = -0.5./q + 0.5*(res.^2)./(q.^2);     % 1×(T-1)×P

Gf = cat(1, dtheta, dq);                      % 2×(T-1)×P
end

% Initial gradient (wrt S0): returns p_init×P with p_init=1
function G0 = grad_log_p1_gauss_vec(X1, init_pars)
% X1 : 1×1×P   (initial state per path)
% G0 : 1×P
S0 = init_pars.S0;
x1 = squeeze(X1);                             % 1×P
G0 = -0.5./S0 + 0.5*(x1.^2)./(S0.^2);        % 1×P
end

%===== scoring helpers (same as earlier) =====

function out = score_gaussian_from_paths_vectorized(y, X_paths, theta, q, r, S0, paths_sel, weights, exclude_first_col)
% Vectorized score over ALL selected paths (no per-path loops).
% y        : 1×T
% X_paths  : 1×T×M×B1
% theta,q,r,S0 : scalars
% paths_sel: logical M×B1 (optional; default all true)
% weights  : P×1 (optional; default uniform over selected)
% exclude_first_col : if true, ignore initializer column (b=1)
%
% Returns:
%   out.avg_obs    (scalar: dr)
%   out.avg_trans  (2×1: [dtheta; dq])
%   out.avg_init   (scalar: dS0)
%   out.avg_total  (4×1: [dr; dtheta; dq; dS0])

    if nargin < 7 || isempty(paths_sel)
        paths_sel = true(size(X_paths,3), size(X_paths,4));
    end
    if nargin >= 9 && exclude_first_col
        paths_sel(:,1) = false;
    end

    [dx,T,M,B1] = size(X_paths); %#ok<ASGLU>
    idx = find(paths_sel(:));
    if isempty(idx), error('No paths selected.'); end
    P = numel(idx);

    % Gather selected paths to 1×T×P
    Xp = zeros(1, T, P, 'like', X_paths);
    [mm, bb] = ind2sub([size(X_paths,3), size(X_paths,4)], idx);
    for k = 1:P
        Xp(:,:,k) = X_paths(:,:,mm(k), bb(k));
    end

    % Weights
    if nargin < 8 || isempty(weights)
        w = ones(P,1, 'like', Xp) / P;
    else
        w = weights(:) / sum(weights);
    end
    wrow = reshape(w, 1, P); %#ok<NASGU>

    % Shapes to T×P
    Xall = squeeze(Xp);            % T×P
    if isrow(Xall), Xall = Xall.'; end
    yrow = y(:).';

    % --- Observation term (dr)
    diff_yx = yrow.' - Xall;                 % T×P
    dr_each = -0.5./r + 0.5*(diff_yx.^2)/(r^2);  % T×P
    dr_k    = sum(dr_each, 1);               % 1×P
    dr      = dr_k * w;                      % scalar

    % --- Transition term (dtheta, dq)
    x_t   = Xall(2:end, :);                  % (T-1)×P
    x_tm1 = Xall(1:end-1, :);                % (T-1)×P
    res   = x_t - theta.*x_tm1;              % (T-1)×P
    dtheta_k = sum( (res .* x_tm1) / q, 1 ); % 1×P
    dq_k     = sum( -0.5./q + 0.5*(res.^2)/(q^2), 1 ); % 1×P
    dtheta   = dtheta_k * w;                 % scalar
    dq       = dq_k * w;                     % scalar

    % --- Initial term (dS0)
    x1 = Xall(1,:);                          % 1×P
    dS0_k = -0.5./S0 + 0.5*(x1.^2)/(S0^2);  % 1×P
    dS0    = dS0_k * w;                     % scalar

    out.avg_obs   = dr;
    out.avg_trans = [dtheta; dq];
    out.avg_init  = dS0;
    out.avg_total = [dr; dtheta; dq; dS0];
end


function trace = sa_pg_gauss_ssm(y,T,N,M,B,K,seed0, ...
                                 theta0,q0,r0,S0, ...
                                 Gamma,alpha,theta_max)
% SA stochastic approximation for Gaussian SSM via PG score.

% Unconstrained params
lq = log(q0); lr = log(r0);
theta = theta0;

trace.theta = zeros(K,1);
trace.q = zeros(K,1);
trace.r = zeros(K,1);
trace.score = zeros(3,K);
trace.step  = zeros(K,1);

for n=1:K
    t0 = tic;

    q = exp(lq); r = exp(lr);

    % --- Build PF/CPF functions ---
    in_pars.mu=0; in_pars.Sigma=S0;
    in_dist=@(p,N_,M_) reshape(p.mu+sqrt(p.Sigma)*randn(1,N_*M_),1,N_,M_);

    tr_pars.theta=theta; tr_pars.q=q; tr_pars.sig=sqrt(q);
    trans=@(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

    g_pars.R=r;
    g=@(yt,Xt,p,t) reshape(-0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R),1,size(Xt,2),size(Xt,3));

    trans_logpdf=@(x_next,X_prev,pars,t) -0.5*((x_next-theta*X_prev).^2)/q - 0.5*log(2*pi*q);

    % --- Run PG to approximate score ---
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

    % --- Convert to unconstrained grads
    grad_theta = g_vec(1);
    grad_lq    = q * g_vec(2);
    grad_lr    = r * g_vec(3);

    % --- Step sizes
    step_theta = Gamma(1)/(100+n)^(alpha+0.5);
    step_q     = Gamma(2)/(100+n)^(alpha+0.5);
    step_r     = Gamma(3)/(100+n)^(alpha+0.5);

    % --- Updates
    theta = theta + step_theta * grad_theta;
    lq    = lq    + step_q     * grad_lq;
    lr    = lr    + step_r     * grad_lr;

    % projection
    theta = max(min(theta, theta_max), -theta_max);

    % store
    trace.theta(n)=theta;
    trace.q(n)=exp(lq);
    trace.r(n)=exp(lr);
    trace.score(:,n)=g_vec;
    trace.step(n)=toc(t0);
end
end

function [unb_est, info] = pg_unbiased_score_gauss_2( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    theta, q, r, S0, ...
    Bs, l_dist, seed0, ...
    cpf_choice, traj_mode, trans_logpdf, ...
    store_particles, store_ancestors, store_logw, ...
    N_init, init_mode)

% Unbiased estimator of [dℓ/dθ; dℓ/dq; dℓ/dr] using randomized PG levels
% for the 1D Gaussian SSM.

    % ---- basic checks ----
    Bs     = Bs(:).';
    l_dist = l_dist(:).';
    Lmax   = numel(Bs) - 1;
    if numel(l_dist) ~= Lmax+1
        error('Bs must have length Lmax+1 and l_dist must have same length.');
    end

    eLes = 0:Lmax;

    % ---- sample level l ~ l_dist ----
    rng(seed0,"twister");
    log_lw = log(l_dist);
    j      = draw_from_logw(log_lw);   % index in 1..Lmax+1
    l      = eLes(j);                  % actual level

    Bs_lvl = Bs(1:l+1);                % cumulative chain lengths for this level
    n_lvl  = numel(Bs_lvl);

    paths_sums = zeros(n_lvl, 3);
    total_cost = 0;                    % simple cost measure: sum of B_inc

    for n = 1:n_lvl
        if n > 1
            B_inc = Bs_lvl(n) - Bs_lvl(n-1);
        else
            B_inc = Bs_lvl(1);
        end

        total_cost = total_cost + B_inc;

        out_pg = pgibbs_run_init_pfmean( ...
            y, T, N, M, B_inc, ...
            in_dist_samp, in_pars, ...
            trans_dist_samp, trans_pars, ...
            g, g_pars, ...
            seed0 + 1000*n, ...
            cpf_choice, ...
            traj_mode, trans_logpdf, ...
            store_particles, store_ancestors, store_logw, ...
            N_init, init_mode);

        X_paths = out_pg.X_paths;

        S = score_gaussian_from_paths_vectorized( ...
                y, X_paths, theta, q, r, S0, ...
                [], [], true);  % use all paths except initializer

        g_vec = [S.avg_trans(1); S.avg_trans(2); S.avg_obs];  % [dθ; dq; dr]

        paths_sums(n,:) = B_inc * g_vec.';   % 1×3
    end

    if l > 0
        num = paths_sums(end,:) - sum(paths_sums(1:end-1,:), 1);
        den = Bs_lvl(end) * l_dist(j);
        unb_row = num / den;
    else
        num = paths_sums(1,:);
        den = Bs_lvl(1) * l_dist(j);
        unb_row = num / den;
    end

    unb_est = unb_row.';  % 3×1

    info.level      = l;
    info.level_idx  = j;
    info.Bs_used    = Bs_lvl;
    info.l_dist     = l_dist;
    info.paths_sums = paths_sums;
    info.seed0      = seed0;
    info.total_cost = total_cost;  % this is what we'll use later
end


function S = score_t_from_paths_vectorized(y, X_paths, init_pars, trans_pars, obs_pars)
%SCORE_T_FROM_PATHS_VECTORIZED
%   Full score for t-observation SSM using *vectorized* init and trans grads.
%
% Inputs:
%   y         : 1×T                (observations)
%   X_paths   : 1×T×P              (P trajectories, e.g. from PG)
%   init_pars : struct, parameters of initial distribution
%   trans_pars: struct, parameters of Gaussian transition
%   obs_pars  : struct, parameters of t-Student obs (e.g. v, sigma)
%
% Required helper functions (vectorized):
%   G_init  = grad_log_p1_gauss_vec(X1, init_pars);
%            % X1: 1×P, G_init: d_init×P  (or 1×P if scalar)
%
%   G_trans = grad_log_f_gauss_vec(X_paths, trans_pars);
%            % X_paths: 1×T×P, G_trans: d_trans×P
%
%   g_obs   = grad_log_g_t_stud_vec(y, X_paths, obs_pars);
%            % y: 1×T, X_paths: 1×T×P, g_obs: 1×P (wrt log sigma)
%
% Output:
%   S : struct with fields
%       .per_path_init   : d_init×P   (or 1×P)
%       .per_path_trans  : d_trans×P
%       .per_path_obs    : 1×P
%       .avg_init        : d_init×1   (mean over paths)
%       .avg_trans       : d_trans×1
%       .avg_obs         : scalar

    [~, T, P] = size(X_paths); %#ok<ASGLU>

    % 1) INITIAL SCORE (vectorized)
    % Extract x_1 for each path: X_paths(1,1,p), p=1..P
    X1 = squeeze(X_paths(1,1,:)).';   % 1×P

    G_init = grad_log_p1_gauss_vec(X1, init_pars);  % d_init×P or 1×P

    % 2) TRANSITION SCORE (vectorized)
    % grad_log_f_gauss_vec handles all t>=2 and all paths internally
    G_trans = grad_log_f_gauss_vec(X_paths, trans_pars);  % d_trans×PxT or d_transx(T-1)xP

    % 3) OBSERVATION SCORE (t-Student, vectorized)
    % This should already be vectorized over t and p
    g_obs = grad_log_g_t_stud_vec(y, X_paths, obs_pars);  % (1×P), notice that
    % the sum wrt to the time is already computed.

    % 4) Pack into output struct
    S.per_path_init   = G_init;          % d_init×P
    S.per_path_trans  = G_trans;         % d_trans×x(T-1)xP
    S.per_path_obs    = g_obs;           % 1×P

    S.avg_init  = mean(G_init,  2);      % d_init×1
    S.avg_trans = squeeze(mean(sum(G_trans, 2),3));      % d_trans×1
    S.avg_obs   = mean(g_obs,   1);      % scalar

end

function S = score_t_from_paths(y, X_paths, init_pars, trans_pars, obs_pars)
%SCORE_T_FROM_PATHS  Full score for t-observation SSM
%
% Computes:
%   - init score
%   - transition score
%   - observation score (t-student, wrt log sigma)
%
% Inputs:
%   y         : 1×T
%   X_paths   : 1×T×P
%   init_pars : parameters of initial state density
%   trans_pars: parameters of Gaussian transition
%   obs_pars  : parameters of t-Student obs: (v, sigma)
%
% Output struct S with fields:
%   .per_path_init
%   .per_path_trans   (vector or matrix depending on parameters)
%   .per_path_obs
%   .avg_init
%   .avg_trans
%   .avg_obs

    [~, T, P] = size(X_paths);

    % ----- 1) INITIAL SCORE (Gaussian) -----
    % grad_log_p1 should return gradient wrt initial parameters
    S_init = zeros(1, P);
    for p = 1:P
        x1 = X_paths(1,1,p);
        S_init(p) = grad_log_p1_gauss(x1, init_pars);
    end

    % ----- 2) TRANSITION SCORE (Gaussian AR(1)) -----
    % grad_log_f(x_t, x_{t-1}, trans_pars)
    S_trans = zeros(2, P);
    for p = 1:P
        x_p = squeeze(X_paths(1,:,p));  % 1×T → row vector
        s = 0;
        for t = 2:T
            s = s + grad_log_f_gauss(x_p(t), x_p(t-1), trans_pars);
        end
        S_trans(:,p) = s;
    end

    % ----- 3) OBSERVATION SCORE (t-Student) -----
    S_obs = grad_log_g_t_stud_vec(y, X_paths, obs_pars);   % 1×P

    % ----- 4) Build output -----
    S.per_path_init  = S_init;
    S.per_path_trans = S_trans;
    S.per_path_obs   = S_obs;

    S.avg_init  = mean(S_init);
    S.avg_trans = mean(S_trans,2);
    S.avg_obs   = mean(S_obs);
end


function unb_est = pg_unbiased_score_tstud_2( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g_t, obs_pars_t, ...
    trans_logpdf, ...
    B0, eLes, l_dist, ...
    seed0, ...
    cpf_choice, traj_mode, ...
    store_particles, store_ancestors, store_logw, ...
    N_init, init_mode)
%PG_UNBIASED_SCORE_TSTUD
%  Unbiased estimator of the score w.r.t. (theta, q, r) using
%  Particle Gibbs with t-Student observations (scale sigma = sqrt(r)).
%
%  The construction mirrors pg_unbiased_score_gauss, but:
%    - observation model is t-Student with df = obs_pars_t.v,
%      scale = obs_pars_t.sigma = sqrt(r);
%    - we only return [dtheta; dq; dr].
%
% Inputs:
%   y              : 1×T observations
%   T              : length
%   N, M           : #particles, #chains per PG iteration
%
%   in_dist_samp   : @(in_pars,N,M) -> 1×N×M initial sampler
%   in_pars        : struct for initial distribution
%
%   trans_dist_samp: @(Xprev,trans_pars,t) -> 1×N×M transition sampler
%   trans_pars     : struct with fields (theta, q, sig=√q)
%
%   g_t            : @(yt,Xt,obs_pars_t,t)->1×N×M  LOG t-likelihood
%   obs_pars_t     : struct with fields
%                       .v     (degrees of freedom)
%                       .sigma (scale parameter = sqrt(r))
%
%   trans_logpdf   : @(x_next, X_prev, trans_pars, t)->1×N
%                    log transition density for backward simulation
%
%   B0             : base number of PG links for level 0
%   eLes           : vector of possible levels, e.g. [0 1 2 ...]
%   l_dist         : probability mass over eLes, same size
%                    (must sum to 1)
%
%   seed0          : base RNG seed (scalar)
%
%   cpf_choice     : "cpf" or "cpf_parallel"
%   traj_mode      : "ancestors" or "backward"
%
%   store_particles, store_ancestors, store_logw : logical flags
%
%   N_init         : #particles used in initial PF in pgibbs_run_init_pfmean
%   init_mode      : e.g. "weighted"
%
% Output:
%   unb_est  : 3×1 column vector [dtheta; dq; dr] (unbiased)
%
% Notes:
%   - Calls pgibbs_run_init_pfmean internally.
%   - Uses score_t_from_paths_vectorized(y, X_paths, in_pars, trans_pars, obs_pars_t)
%     to compute:
%         S.avg_trans(1) = dtheta
%         S.avg_trans(2) = dq
%         S.avg_obs      = d/d(log sigma)
%     and then transforms avg_obs -> dr via
%         dr = avg_obs / (2*r),  r = obs_pars_t.sigma^2

    % --- input checks for l_dist ---
    if numel(eLes) ~= numel(l_dist)
        error('eLes and l_dist must have the same length.');
    end
    if abs(sum(l_dist) - 1) > 1e-10
        warning('l_dist does not sum to 1. Normalizing.');
        l_dist = l_dist(:).' / sum(l_dist);
    else
        l_dist = l_dist(:).';  % row
    end

    % --- draw a random level L from {eLes} with pmf l_dist ---
    % use log-weights sampling for numerical stability
    logw_levels = log(l_dist);
    rng(seed0,'twister');
    j = draw_from_logw(logw_levels);    % index in 1..numel(eLes)
    L = eLes(j);

    % --- build B-grid: Bs = B0*2.^(0:L) ---
    Bs = B0 * 2.^(0:L);
    nLevels = numel(Bs);

    % --- allocate accumulator over levels: each row is a level ---
    % we store [dtheta; dq; dr] * B
    paths_sums = zeros(nLevels, 3);

    % For convenience
    r_current = obs_pars_t.sigma^2;

    % --- loop over levels n=1..nLevels ---
    for n = 1:nLevels

        if n == 1
            B = Bs(1);             % first level: B links
        else
            B = Bs(n) - Bs(n-1);   % incremental links for level n
        end

        % --- run PG with B extra links ---
        out_pg = pgibbs_run_init_pfmean( ...
            y, T, N, M, B, ...
            in_dist_samp, in_pars, ...
            trans_dist_samp, trans_pars, ...
            g_t, obs_pars_t, ...
            seed0 + 1000*n, ...
            cpf_choice, traj_mode, trans_logpdf, ...
            store_particles, store_ancestors, store_logw, ...
            N_init, init_mode);

        % out_pg.X_paths: 1×T×M×(B+1), flatten to 1×T×P
        X_paths_all = out_pg.X_paths;
        X_paths_all=X_paths_all(:,:,:,2:end);
        [~, ~, M_, Bp1] = size(X_paths_all);
        P = M_ * Bp1;
        X_paths = reshape(X_paths_all, 1, T, P);  % 1×T×P

        % --- compute score from these paths (t-Student obs) ---
        S = score_t_from_paths_vectorized(y, X_paths, in_pars, trans_pars, obs_pars_t);

        % Transition components: [dtheta; dq]
        dtheta = S.avg_trans(1);
        dq     = S.avg_trans(2);

        % Observation component: avg_obs is dℓ/dlogσ
        dlog_sigma = S.avg_obs;

        % Convert to dℓ/dr, using r = sigma^2:
        %   dℓ/dr = (1/(2r)) * dℓ/dlogσ
        dr = dlog_sigma / (2 * r_current);

        % Put into vector [dtheta; dq; dr]
        g_vec = [dtheta; dq; dr];  % 3×1

        % Store B * g_vec as one row
        paths_sums(n,:) = (B * g_vec).';  % 1×3 row
    end

    % --- Rhee-Glynn debiasing ---
    if L > 0
        % L corresponds to Bs(end)
        numerator = paths_sums(end,:) - sum(paths_sums(1:end-1,:), 1);
        denom     = Bs(end) * l_dist(j);
        unb_est_row = numerator / denom;   % 1×3
    else
        % level 0 case
        numerator = paths_sums(1,:);
        denom     = Bs(1) * l_dist(j);
        unb_est_row = numerator / denom;   % 1×3
    end

    % Return as 3×1 column: [dtheta; dq; dr]
    unb_est = unb_est_row(:);
end


function trace = sa_pg_unbiased_mixture_clean( ...
    theta0, q0, r0, ...               % initial physical parameters
    K_SA, Gamma, alpha, n0, ...       % SA steps & schedule
    m_mix, S, ...                     % mixture parameter & #unb draws per iter
    unb_gauss, unb_tstud, ...         % @(theta,q,r,seed)->[dθ; dq; dr]
    seed0)

    % Allocate traces in (θ, log q, log r)
    theta_trace = zeros(K_SA+1,1);
    lq_trace    = zeros(K_SA+1,1);
    lr_trace    = zeros(K_SA+1,1);
    grad_trace  = zeros(K_SA,3);
    gamma_trace = zeros(K_SA,3);
    family_trace= zeros(K_SA,1); % 1=Gaussian,2=t-Student

    % Init in transformed domain
    theta_trace(1) = theta0;
    lq_trace(1)    = log(q0);
    lr_trace(1)    = log(r0);

    Gamma = Gamma(:);   % 3×1
    rng(seed0,"twister");

    p_gauss = m_mix/(m_mix+1);

    for n = 1:K_SA

        % Current SA state
        theta_n = theta_trace(n);
        lq_n    = lq_trace(n);
        lr_n    = lr_trace(n);

        % Physical params
        q_n = exp(lq_n);
        r_n = exp(lr_n);

        % Step sizes
        gamma_n = Gamma ./ ( (n0 + n)^(alpha + 0.5) );
        gamma_trace(n,:) = gamma_n.';

        % Choose family
        if rand < p_gauss
            fam = 1;  % Gaussian
        else
            fam = 2;  % t-Stud
        end
        family_trace(n) = fam;

        % Collect S unbiased draws
        G_phys = zeros(3,S);   % in (θ,q,r)
        for s = 1:S
            seed_s = seed0 + 100000*n + 1000*s;
            if fam == 1
                G_phys(:,s) = unb_gauss(theta_n, q_n, r_n, seed_s);
            else
                G_phys(:,s) = unb_tstud(theta_n, q_n, r_n, seed_s);
            end
        end

        % Average & transform to (θ, log q, log r)
        U_raw = mean(G_phys,2);   % [dθ; dq; dr]

        g_hat = zeros(3,1);
        g_hat(1) = U_raw(1);
        g_hat(2) = U_raw(2) * q_n;  % d/d(log q)
        g_hat(3) = U_raw(3) * r_n;  % d/d(log r)

        grad_trace(n,:) = g_hat.';

        % SA update in transformed domain
        theta_next = theta_n + gamma_n(1)*g_hat(1);
        lq_next    = lq_n    + gamma_n(2)*g_hat(2);
        lr_next    = lr_n    + gamma_n(3)*g_hat(3);

        theta_trace(n+1) = theta_next;
        lq_trace(n+1)    = lq_next;
        lr_trace(n+1)    = lr_next;
    end

    % Pack output, both transformed and physical
    trace.theta  = theta_trace;
    trace.q      = exp(lq_trace);
    trace.r      = exp(lr_trace);
    trace.lq     = lq_trace;
    trace.lr     = lr_trace;
    trace.grad   = grad_trace;
    trace.gamma  = gamma_trace;
    trace.family = family_trace;
    trace.S      = S;
    trace.m_mix  = m_mix;
end


function [unb_est, info] = pg_unbiased_score_gauss( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    theta, q, r, S0, ...          % physical parameters
    Bs, l_dist, seed0, ...        % debiasing parameters
    cpf_choice, traj_mode, trans_logpdf, ...
    store_particles, store_ancestors, store_logw, ...
    N_init, init_mode)

    % Consistent transition pars
    trans_pars.theta = theta;
    trans_pars.q     = q;
    trans_pars.sig   = sqrt(q);

    % Observation pars
    g_pars.R = r;

    % ---- check l_dist ----
    Bs     = Bs(:).';
    l_dist = l_dist(:).';
    Lmax   = numel(Bs) - 1;
    if numel(l_dist) ~= Lmax+1
        error('Bs and l_dist lengths mismatch.');
    end

    eLes = 0:Lmax;

    % ---- sample level ----
    rng(seed0,'twister');
    j = draw_from_logw(log(l_dist));
    L = eLes(j);

    Bs_lvl = Bs(1:L+1);
    n_lvl  = numel(Bs_lvl);

    paths_sums = zeros(n_lvl, 3);

    % ---- Loop over levels ----
    for n = 1:n_lvl

        if n==1
            B_inc = Bs_lvl(1);
        else
            B_inc = Bs_lvl(n)-Bs_lvl(n-1);
        end

        out_pg = pgibbs_run_init_pfmean( ...
            y, T, N, M, B_inc, ...
            in_dist_samp, in_pars, ...
            trans_dist_samp, trans_pars, ...
            g, g_pars, ...
            seed0 + 1000*n, ...
            cpf_choice, traj_mode, trans_logpdf, ...
            store_particles, store_ancestors, store_logw, ...
            N_init, init_mode);

        X_paths = out_pg.X_paths;

        S = score_gaussian_from_paths_vectorized( ...
            y, X_paths, theta, q, r, S0, [], [], true);

        g_vec = [S.avg_trans(1); S.avg_trans(2); S.avg_obs];  % [dθ; dq; dr]

        paths_sums(n,:) = B_inc * g_vec.';
    end

    % ---- Rhee & Glynn unbiased assembly ----
    if L > 0
        num = paths_sums(end,:) - sum(paths_sums(1:end-1,:),1);
        den = Bs_lvl(end) * l_dist(j);
        unb_est = (num/den).';
    else
        num = paths_sums(1,:);
        den = Bs_lvl(1) * l_dist(j);
        unb_est = (num/den).';
    end

    % info
    info.level      = L;
    info.level_idx  = j;
    info.paths_sums = paths_sums;
    info.Bs         = Bs_lvl;
end

function unb_est = pg_unbiased_score_tstud( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g_t, obs_pars_t, ...
    trans_logpdf, ...
    B0, eLes, l_dist, seed0, ...
    cpf_choice, traj_mode, ...
    store_particles, store_ancestors, store_logw, ...
    N_init, init_mode)

    % Consistent transition pars
    theta = trans_pars.theta;
    q     = trans_pars.q;
    trans_pars.sig = sqrt(q);

    % Observation scale
    sigma = obs_pars_t.sigma;  % sqrt(r)
    r     = sigma^2;

    % normalize l_dist
    l_dist = l_dist(:).'/sum(l_dist);
    eLes   = eLes(:).';

    rng(seed0,'twister');
    j = draw_from_logw(log(l_dist));
    L = eLes(j);

    Bs = B0 * 2.^(0:L);
    nLevels = numel(Bs);
    paths_sums = zeros(nLevels, 3);

    for n = 1:nLevels
        if n==1, B = Bs(1); 
        else,   B = Bs(n)-Bs(n-1);
        end

        out_pg = pgibbs_run_init_pfmean( ...
            y, T, N, M, B, ...
            in_dist_samp, in_pars, ...
            trans_dist_samp, trans_pars, ...
            g_t, obs_pars_t, ...
            seed0+1000*n, ...
            cpf_choice, traj_mode, trans_logpdf, ...
            store_particles, store_ancestors, store_logw, ...
            N_init, init_mode);





        X_paths_all = out_pg.X_paths;
        X_paths_all=X_paths_all(:,:,:,2:end);
        [~,T2,M2,Bp1] = size(X_paths_all);
        P = M2*Bp1;
        X_paths = reshape(X_paths_all,1,T2,P);

        S = score_t_from_paths_vectorized(y, X_paths, in_pars, trans_pars, obs_pars_t);

        dtheta = S.avg_trans(1);
        dq     = S.avg_trans(2);

        dlog_sigma = S.avg_obs;       % derivative wrt log σ
        dr = dlog_sigma/(2*r);        % chain rule

        g_vec = [dtheta; dq; dr];
        paths_sums(n,:) = B*g_vec.';
    end

    if L>0
        unb_est = (paths_sums(end,:) - sum(paths_sums(1:end-1,:),1)) / (Bs(end)*l_dist(j));
    else
        unb_est = paths_sums(1,:) / (Bs(1)*l_dist(j));
    end
    unb_est = unb_est(:);
end


function [score, parts, loglik] = score_gaussian_ssm_new(y, theta, q, r, S0)
% SCORE_GAUSSIAN_SSM  Gradient (score) of the log-likelihood for a 1D LGSSM.
% Model:
%   X1 ~ N(0, S0)
%   Xt | X_{t-1} ~ N(theta * X_{t-1}, q)
%   Yt | Xt ~ N(Xt, r)
%
% Inputs
%   y      : 1×T observations
%   theta  : scalar (state coefficient)
%   q      : scalar > 0 (state var)
%   r      : scalar > 0 (obs var)
%   S0     : scalar > 0 (initial var)
%
% Outputs
%   score  : struct with fields dtheta, dq, dr, dS0
%   parts  : struct with useful internals (filtered, smoothed, etc.)
%   loglik : scalar log-likelihood log p(y | theta,q,r,S0)

    y  = y(:)';                 % 1×T
    T  = size(y,2);
    eps_small = 1e-12;

    % ---------- Forward: Kalman filter (update at t, then predict to t+1)
    m_f = zeros(1,T);  P_f = zeros(1,T);
    m_pred = zeros(1,T);  P_pred = zeros(1,T);   % store P_{t|t-1}
    % prior for t=1 (before seeing y1)
    m = 0;
    P = S0;

    % log-likelihood accumulator
    loglik = 0;
    log2pi = log(2*pi);

    for t = 1:T
        % innovation update at t
        S_t = P + r;                          % 1×1
        K   = P / S_t;
        innov = y(t) - m;

        % contribution to log-likelihood
        loglik = loglik - 0.5*( log2pi + log(S_t) + (innov^2)/S_t );

        % update state posterior
        m = m + K * innov;
        P = (1 - K) * P;
        m_f(t) = m;  P_f(t) = P;

        % prediction to t+1
        if t < T
            m_pred(t+1) = theta * m;
            P_pred(t+1) = theta^2 * P + q;
            m = m_pred(t+1);
            P = P_pred(t+1);
        end
    end

    % ---------- Backward: RTS smoother + lag-one covariance
    m_s = zeros(1,T);  P_s = zeros(1,T);
    J   = zeros(1,T-1);           % smoother gains J_t for t=1..T-1
    C_lag = zeros(1,T);           % Cov(X_t, X_{t-1} | y), t>=2

    m_s(T) = m_f(T);  P_s(T) = P_f(T);
    for t = T-1:-1:1
        % P_pred(t+1) is prediction variance from t to t+1
        Pp = max(P_pred(t+1), eps_small);
        J(t) = (P_f(t) * theta) / Pp;

        % smooth
        m_s(t) = m_f(t) + J(t) * (m_s(t+1) - theta*m_f(t));
        P_s(t) = P_f(t) + J(t)^2 * (P_s(t+1) - Pp);
    end
    for t = 2:T
        C_lag(t) = J(t-1) * P_s(t);   % Cov(X_t, X_{t-1} | y)
    end

    % ---------- Expectations needed for the score
    EX2  = P_s + m_s.^2;
    EXXt = C_lag + m_s .* [0, m_s(1:end-1)];

    % (1) wrt theta
    E_XtXm1   = EXXt(2:end);
    E_Xm1sq   = EX2(1:end-1);
    dtheta = (1/max(q,eps_small)) * sum( E_XtXm1 - theta * E_Xm1sq );

    % (2) wrt q
    E_res2 = EX2(2:end) - 2*theta*E_XtXm1 + theta^2 * E_Xm1sq;
    dq = -(T-1)/(2*max(q,eps_small)) + 0.5 * sum(E_res2) / max(q,eps_small)^2;

    % (3) wrt r
    E_meas2 = (y - m_s).^2 + P_s;
    dr = -T/(2*max(r,eps_small)) + 0.5 * sum(E_meas2) / max(r,eps_small)^2;

    % (4) wrt S0
    dS0 = -1/(2*max(S0,eps_small)) + 0.5 * EX2(1) / max(S0,eps_small)^2;

    % ---------- Package
    score = struct('dtheta', dtheta, 'dq', dq, 'dr', dr, 'dS0', dS0);

    if nargout > 1
        parts = struct();
        parts.m_f    = m_f;    parts.P_f    = P_f;
        parts.m_s    = m_s;    parts.P_s    = P_s;
        parts.J      = J;      parts.P_pred = P_pred;
        parts.C_lag  = C_lag;
        parts.EX2    = EX2;    parts.EXXt   = EXXt;
        parts.E_res2 = E_res2; parts.E_meas2 = E_meas2;
        parts.loglik = loglik;  % <-- store here too
    end
end

function ll = g_mix_t_gauss(yt, Xt, pars, t)
%G_MIX_T_GAUSS  Log-likelihood for a mixture of t-Student and Gaussian.
%
%   ll = g_mix_t_gauss(yt, Xt, pars, t)
%
% Inputs:
%   yt   : 1×1   (or 1×d_y, d_y=1) observation at time t
%   Xt   : 1×N×M particles at time t
%   pars : struct with fields
%          .R      : Gaussian observation variance r
%          .v      : t-Student degrees of freedom nu
%          .sigma  : t-Student scale sigma
%          .m_mix  : mixing parameter m (scalar)
%                    weight_t   = 1/(m_mix+1)
%                    weight_gauss = m_mix/(m_mix+1)
%   t    : time index (unused here, but kept for interface consistency)
%
% Output:
%   ll   : 1×N×M log-likelihood of the mixture

    R     = pars.R;
    v     = pars.v;
    sigma = pars.sigma;
    m_mix = pars.m_mix;
    
        % ---- Check whether sigma^2 and R are (numerically) equal ----
    tol = 1e-12;
    same_var = abs(sigma^2 - R) <= tol * max(1, abs(R));

    if same_var
        % Optional: display a notice for debugging once
         persistent warned
         if isempty(warned)
             warning('g\_mix\_t\_gauss: sigma^2 and R are numerically equal.');
             warned = true;
         end
    end

    % ----- residuals -----
    % Xt is 1×N×M, yt is 1×1 ⇒ broadcast to 1×N×M
    diff = yt - Xt;      % 1×N×M

    % ----- Gaussian log-density -----
    % N(Xt, R)
    ll_gauss = -0.5 * (diff.^2) / R ...
               - 0.5 * log(2*pi*R);     % 1×N×M

    % ----- t-Student log-density (1D) -----
    % f(y|x) = c * ( 1 + (diff^2)/(v*sigma^2) )^{-(v+1)/2}
    z = (diff ./ sigma).^2;             % 1×N×M

    const_t = gammaln((v+1)/2) ...
              - gammaln(v/2) ...
              - 0.5*log(v*pi) ...
              - log(sigma);

    ll_t = const_t ...
           - 0.5*(v+1) .* log(1 + z./v);   % 1×N×M

    % ----- mixture weights -----
    w_t = 1/(m_mix + 1);
    w_g = m_mix/(m_mix + 1);

    log_w_t = log(w_t);
    log_w_g = log(w_g);

    A = ll_t     + log_w_t;   % 1×N×M
    B = ll_gauss + log_w_g;   % 1×N×M

    % ----- log-sum-exp for mixture -----
    Lmax = max(A, B);                         % 1×N×M
    ll = Lmax + log( exp(A - Lmax) + exp(B - Lmax) );  % 1×N×M
end



function S = score_mix_from_paths_vectorized( ...
    y, X_paths, init_pars, trans_pars, obs_pars_mix)
%SCORE_MIX_FROM_PATHS_VECTORIZED
%  Full score for *mixture* observation SSM using vectorized init/transition grads.
%
% Model pieces:
%   - Initial: X1 ~ N(0, S0)
%   - Transition: Xt|X_{t-1} ~ N(theta * X_{t-1}, q)
%   - Observation (mixture):
%       p(y_t|x_t) = w_t f_t(y_t|x_t; v, sigma) + w_g f_g(y_t|x_t; r)
%
% Inputs:
%   y            : 1×T
%   X_paths      : 1×T×P
%   init_pars    : struct with fields, e.g. init_pars.S0
%   trans_pars   : struct with fields theta, q (and maybe sig = sqrt(q))
%   obs_pars_mix : struct with fields:
%                    .R      : Gaussian variance r
%                    .v      : t-Student df
%                    .sigma  : t-Student scale
%                    .m_mix  : mixing param m
%
% Required helper functions (vectorized, as before):
%   G_init  = grad_log_p1_gauss_vec(X1, init_pars);
%   G_trans = grad_log_f_gauss_vec(X_paths, trans_pars);
%
%   [log_ft, dlogft_dlogsigma] = t_component_log_and_grad(y, X_paths, v, sigma)
%   [log_fg, dlogfg_dr]        = gauss_component_log_and_grad(y, X_paths, r)
%
% Output:
%   S struct:
%     .per_path_init   : d_init×P
%     .per_path_trans  : d_trans×P
%     .per_path_obs    : 1×P      (sum over time of dℓ/d r for each path)
%     .avg_init        : d_init×1
%     .avg_trans       : d_trans×1
%     .avg_obs         : scalar

    [~, T, P] = size(X_paths); %#ok<ASGLU>

    % 1) Initial score (Gaussian, vectorized)
    X1 = squeeze(X_paths(1,1,:)).';             % 1×P
    G_init = grad_log_p1_gauss_vec(X1, init_pars);  % d_init×P

    % 2) Transition score (Gaussian AR(1), vectorized)
    G_trans = grad_log_f_gauss_vec(X_paths, trans_pars); % 2×(T-1)×P
    % Sum over time and average over paths later.

    % 3) Observation score for *mixture* wrt r
    R     = obs_pars_mix.R;
    v     = obs_pars_mix.v;
    sigma = obs_pars_mix.sigma;
    m_mix = obs_pars_mix.m_mix;

    w_t = 1/(m_mix + 1);   % t weight in mixture
    w_g = m_mix/(m_mix+1); % Gaussian weight

    % --- residuals y - x (broadcast) ---
    diff = y - X_paths;         % 1×T×P
    z    = diff.^2;             % 1×T×P

    % --- Gaussian component: N(x_t, R) ---
    log_fg_tp = -0.5*log(2*pi*R) - 0.5*z./R;     % 1×T×P
    % d/dR log f_g = -1/(2R) + (z)/(2R^2)
    dlogfg_dR_tp = -0.5./R + 0.5*z./(R^2);       % 1×T×P

    % --- t-Student component: location x_t, scale sigma ---
    % log f_t(y|x) = const - log(sigma) - 0.5(v+1) log(1 + z/(v*sigma^2))
    const_t = gammaln((v+1)/2) - gammaln(v/2) ...
              - 0.5*log(v*pi) - log(sigma);
    log_ft_tp = const_t - 0.5*(v+1).*log(1 + z./(v*sigma^2));  % 1×T×P

    % We suppose we *already derived* the derivative wrt log(sigma):
    %   d/d log(sigma) log f_t = -1 + (v+1) * z / (v*sigma^2 + z)
    denom = v*sigma^2 + z;                          % 1×T×P
    dlogft_dlogsigma_tp = -1 + (v+1).* (z ./ denom);% 1×T×P

    % If we want derivative wrt r = sigma^2, use chain rule:
    %   log(sigma) = 0.5 log(r) ⇒ d/d r log(sigma) = 1/(2r)
    %   d/d r log f_t = (1/(2r)) * d/d log(sigma) log f_t
    r_equiv = sigma^2;    % if you want to think in terms of r
    dlogft_dR_tp = (1/(2*r_equiv)) * dlogft_dlogsigma_tp;  % 1×T×P

    % --- mixture: p_t = w_t f_t + w_g f_g ---
    % Work in log domain for stability:
    log_w_t = log(w_t);
    log_w_g = log(w_g);

    log_num_t = log_w_t + log_ft_tp;   % 1×T×P
    log_num_g = log_w_g + log_fg_tp;   % 1×T×P

    % log_den = log( w_t f_t + w_g f_g ) = logsumexp(log_num_t, log_num_g)
    max_l = max(log_num_t, log_num_g);
    % safe log-sum-exp
    log_den = max_l + log( exp(log_num_t - max_l) + exp(log_num_g - max_l) );  % 1×T×P

    % posterior responsibilities α, β
    alpha_tp = exp(log_num_t - log_den);   % 1×T×P
    beta_tp  = exp(log_num_g - log_den);   % 1×T×P

    % final derivative wrt R (the obs variance parameter) per time & path:
    dlogp_dR_tp = alpha_tp .* dlogft_dR_tp + beta_tp .* dlogfg_dR_tp;  % 1×T×P

    % Sum over time for each path → 1×P
    per_path_obs = squeeze(sum(dlogp_dR_tp, 2)).';   % 1×P

    % 4) Pack results
    S.per_path_init   = G_init;                     % d_init×P
    % For transitions, you likely want sum over t → 2×P
    S.per_path_trans  = squeeze(sum(G_trans, 2));   % 2×P
    S.per_path_obs    = per_path_obs;              % 1×P

    S.avg_init  = mean(G_init,  2);                % d_init×1
    S.avg_trans = mean(S.per_path_trans, 2);       % 2×1
    S.avg_obs   = mean(per_path_obs, 2);           % scalar
end

function trace = sa_pg_mix_ssm( ...
    y, T, ...
    N, M, B, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars0, ...
    g_mix, obs_pars_mix0, ...
    seed0, ...
    cpf_choice, traj_mode, trans_logpdf, ...
    store_particles, store_ancestors, store_logw, ...
    N_init, init_mode, ...
    K, alpha, Gamma, ...
    theta_max)
%SA_PG_MIX_SSM  SA for 1D LGSSM with *mixture* observations using PG.
%
%   trace = sa_pg_mix_ssm(y, T, ...
%       N, M, B, ...
%       in_dist_samp, in_pars, ...
%       trans_dist_samp, trans_pars0, ...
%       g_mix, obs_pars_mix0, ...
%       seed0, ...
%       cpf_choice, traj_mode, trans_logpdf, ...
%       store_particles, store_ancestors, store_logw, ...
%       N_init, init_mode, ...
%       K, alpha, Gamma, ...
%       theta_max)
%
% Inputs:
%   y         : 1×T observations
%   T         : length of time series
%
%   N, M      : #particles, #chains per PG iteration
%   B         : #PG links per SA step (fixed for this SA run)
%
%   in_dist_samp : @(in_pars,N,M)->1×N×M  initial sampler
%   in_pars      : struct, with at least field S0 (initial variance)
%
%   trans_dist_samp : @(Xprev, trans_pars, t)->1×N×M
%   trans_pars0     : struct with initial state params:
%                       .theta  (scalar)
%                       .q      (scalar)
%                       .sig    (sqrt(q) or will be overwritten)
%
%   g_mix          : @(y_t, X_t, obs_pars_mix, t)->1×N×M   (LOG mixture likelihood)
%   obs_pars_mix0  : struct with fields:
%                     .R      : Gaussian variance r
%                     .v      : t-Student df
%                     .sigma  : t-Student scale (will be tied to sqrt(r))
%                     .m_mix  : mixing param m
%
%   seed0          : base RNG seed (scalar)
%
%   cpf_choice     : "cpf" or "cpf_parallel"
%   traj_mode      : "ancestors" or "backward"
%   trans_logpdf   : @(x_next, X_prev, trans_pars, t)->1×N  LOG transition density
%
%   store_particles, store_ancestors, store_logw : logical flags
%
%   N_init         : #particles used in initial PF inside pgibbs_run_init_pfmean
%   init_mode      : e.g. "weighted"
%
%   K              : #SA iterations
%   alpha          : exponent in step size schedule
%                    gamma_n = Gamma ./ (100+n)^(alpha+0.5)
%   Gamma          : 3×1 vector of base step sizes for [theta; log q; log r]
%   theta_max      : scalar > 0, projection bound for |theta|
%
% Output:
%   trace : struct with fields
%       .theta   : (K+1)×1  iterates
%       .logq    : (K+1)×1
%       .logr    : (K+1)×1
%       .q       : (K+1)×1
%       .r       : (K+1)×1
%       .g_theta : K×1   (score component dℓ/dθ per step)
%       .g_q     : K×1   (physical dℓ/dq per step)
%       .g_r     : K×1   (physical dℓ/dr per step)
%       .g_logq  : K×1   (score component in log q)
%       .g_logr  : K×1   (score component in log r)
%       .g_norm  : K×1   (norm of gradient in (θ,logq,logr)-space)
%
% Notes:
%   - Observations follow a mixture model
%       p(y_t|x_t) = w_t f_t(y_t|x_t; v, sigma) + w_g f_g(y_t|x_t; R)
%     where w_t = 1/(m_mix+1), w_g = m_mix/(m_mix+1).
%   - This SA is *not* unbiased: each gradient is computed from a fixed-length
%     Particle Gibbs run with mixture observations, averaged over the paths.

    % ---------- 0) Basic checks / defaults ----------
    if nargin < 29 || isempty(theta_max)
        theta_max = 0.999;   % safe AR(1) stability bound
    end

    % flatten y to 1×T just in case
    y = y(:).';
    if length(y) ~= T
        error('Length of y (%d) does not match T=%d.', length(y), T);
    end

    % Transition initial parameters
    theta0 = trans_pars0.theta;
    q0     = trans_pars0.q;
    if q0 <= 0
        error('Initial q0 must be > 0.');
    end

    % Observation initial variance
    r0 = obs_pars_mix0.R;
    if r0 <= 0
        error('Initial R (obs variance) must be > 0.');
    end

    % ---------- 1) Initialize parameters in (theta, log q, log r) ----------
    theta_n = theta0;
    logq_n  = log(q0);
    logr_n  = log(r0);

    % Storage for trace
    trace.theta  = zeros(K+1,1);
    trace.logq   = zeros(K+1,1);
    trace.logr   = zeros(K+1,1);
    trace.q      = zeros(K+1,1);
    trace.r      = zeros(K+1,1);

    trace.g_theta = zeros(K,1);
    trace.g_q     = zeros(K,1);
    trace.g_r     = zeros(K,1);
    trace.g_logq  = zeros(K,1);
    trace.g_logr  = zeros(K,1);
    trace.g_norm  = zeros(K,1);

    % set initial trace
    trace.theta(1) = theta_n;
    trace.logq(1)  = logq_n;
    trace.logr(1)  = logr_n;
    trace.q(1)     = exp(logq_n);
    trace.r(1)     = exp(logr_n);

    % ---------- 2) SA loop ----------
    for n = 1:K

        % Recover physical variances
        q_n = exp(logq_n);
        r_n = exp(logr_n);

        % (a) Build current transition parameters
        trans_pars = trans_pars0;   % copy structure, overwrite fields
        trans_pars.theta = theta_n;
        trans_pars.q     = q_n;
        trans_pars.sig   = sqrt(q_n);

        % (b) Build current mixture obs parameters
        obs_pars_mix = obs_pars_mix0;     % copy fixed fields (v, m_mix)
        obs_pars_mix.R     = r_n;
        obs_pars_mix.sigma = sqrt(r_n);   % tie t-scale to sqrt(r)

        % (c) Run PG (cpf or cpf_parallel) with B links
        seed_n = seed0 + 1000*n;
        out_pg = pgibbs_run_init_pfmean( ...
            y, T, N, M, B, ...
            in_dist_samp, in_pars, ...
            trans_dist_samp, trans_pars, ...
            g_mix, obs_pars_mix, ...
            seed_n, ...
            cpf_choice, traj_mode, trans_logpdf, ...
            store_particles, store_ancestors, store_logw, ...
            N_init, init_mode);

        % out_pg.X_paths: 1×T×M×(B+1), flatten to 1×T×P
        X_paths_all = out_pg.X_paths(:,:,:,1:end-1);
        [~, T_chk, M_chk, Bp1] = size(X_paths_all);
        if T_chk ~= T
            error('X_paths_all has T=%d, expected %d.', T_chk, T);
        end
        P = M_chk * Bp1;
        X_paths = reshape(X_paths_all, 1, T, P);

        % (d) Compute mixture score from these paths
        %     score_mix_from_paths_vectorized should return:
        %       S.avg_trans(1) ≈ dℓ/dθ
        %       S.avg_trans(2) ≈ dℓ/dq
        %       S.avg_obs      ≈ dℓ/dR   (R = r_n)
        S_mix = score_mix_from_paths_vectorized( ...
                    y, X_paths, in_pars, trans_pars, obs_pars_mix);

        dtheta_phys = S_mix.avg_trans(1);   % dℓ/dθ
        dq_phys     = S_mix.avg_trans(2);   % dℓ/dq
        dR_phys     = S_mix.avg_obs;        % dℓ/dR

        % (e) Map to gradient in (θ, log q, log r) coordinates
        %     d/d log q = q * d/dq
        %     d/d log r = r * d/dR
        g_theta = dtheta_phys;
        g_logq  = q_n * dq_phys;
        g_logr  = r_n * dR_phys;

        % Store physical grads too (optional)
        trace.g_theta(n) = g_theta;
        trace.g_q(n)     = dq_phys;
        
        trace.g_r(n)     = dR_phys;
        trace.g_logq(n)  = g_logq;
        trace.g_logr(n)  = g_logr;
        trace.g_norm(n)  = norm([g_theta; g_logq; g_logr]);

        % (f) Step sizes: γ_n = Gamma ./ (100 + n)^(α + 0.5)
        step_scalar = (100 + n)^(alpha + 0.5);
        gamma_n_vec = Gamma(:) ./ step_scalar;   % 3×1

        % (g) SA update in (θ, logq, logr)
        theta_n = theta_n + gamma_n_vec(1)*g_theta;
        logq_n  = logq_n  + gamma_n_vec(2)*g_logq;
        logr_n  = logr_n  + gamma_n_vec(3)*g_logr;

        % (h) Projection / constraints
        if abs(theta_n) > theta_max
            theta_n = sign(theta_n) * theta_max;
        end

        % (i) store new iterate
        trace.theta(n+1) = theta_n;
        trace.logq(n+1)  = logq_n;
        trace.logr(n+1)  = logr_n;
        trace.q(n+1)     = exp(logq_n);
        trace.r(n+1)     = exp(logr_n);
    end
end

function trace = sa_pg_unbiased_model_average_clean( ...
    theta0, q0, r0, ...               % initial physical parameters
    K_SA, Gamma, alpha, n0, ...       % SA steps & schedule
    m_star, S, ...                    % model space size & #unb draws per iter
    unb_gauss, unb_tstud, ...         % gauss: @(theta,q,r,seed)
    seed0)                                 % tstud: @(theta,q,r,nu,seed)
    

    % Allocate traces in (theta, log q, log r)
    theta_trace = zeros(K_SA+1,1);
    lq_trace    = zeros(K_SA+1,1);
    lr_trace    = zeros(K_SA+1,1);
    grad_trace  = zeros(K_SA,3);
    gamma_trace = zeros(K_SA,3);

    % Model trace:
    % M = 1,...,m_star-1 means Student-t with df M
    % M = m_star means Gaussian
    model_trace = zeros(K_SA,1);

    % Init in transformed domain
    theta_trace(1) = theta0;
    lq_trace(1)    = log(q0);
    lr_trace(1)    = log(r0);

    Gamma = Gamma(:);   % 3 x 1
    rng(seed0,"twister");

    for n = 1:K_SA

        % Current SA state
        theta_n = theta_trace(n);
        lq_n    = lq_trace(n);
        lr_n    = lr_trace(n);

        % Physical parameters
        q_n = exp(lq_n);
        r_n = exp(lr_n);

        % Step sizes
        gamma_n = Gamma ./ ((n0 + n)^(alpha + 0.5));
        gamma_trace(n,:) = gamma_n.';

        % Sample model index uniformly from {1,...,m_star}
        M = randi(m_star);
        model_trace(n) = M;

        % Collect S unbiased draws
        G_phys = zeros(3,S);   % in (theta,q,r)

        for s = 1:S
            seed_s = seed0 + 100000*n + 1000*s;

            if M == m_star
                % Gaussian observation model
                G_phys(:,s) = unb_gauss(theta_n, q_n, r_n, seed_s);
            else
                % Student-t observation model with df = M
                nu = M;
                G_phys(:,s) = unb_tstud(theta_n, q_n, r_n, nu, seed_s);
            end
        end

        % Average score in physical parameters
        U_raw = mean(G_phys,2);   % [dtheta; dq; dr]

        % Transform score to (theta, log q, log r)
        g_hat = zeros(3,1);
        g_hat(1) = U_raw(1);
        g_hat(2) = U_raw(2) * q_n;
        g_hat(3) = U_raw(3) * r_n;

        grad_trace(n,:) = g_hat.';

        % SA update in transformed domain
        theta_trace(n+1) = theta_n + gamma_n(1)*g_hat(1);
        lq_trace(n+1)    = lq_n    + gamma_n(2)*g_hat(2);
        lr_trace(n+1)    = lr_n    + gamma_n(3)*g_hat(3);
    end

    % Pack output
    trace.theta = theta_trace;
    trace.q     = exp(lq_trace);
    trace.r     = exp(lr_trace);
    trace.lq    = lq_trace;
    trace.lr    = lr_trace;
    trace.grad  = grad_trace;
    trace.gamma = gamma_trace;

    trace.model = model_trace;
    trace.S     = S;
    trace.m_star = m_star;
end