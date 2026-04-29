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
