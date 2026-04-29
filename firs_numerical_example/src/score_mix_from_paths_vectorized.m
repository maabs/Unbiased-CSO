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

