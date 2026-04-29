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


