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

