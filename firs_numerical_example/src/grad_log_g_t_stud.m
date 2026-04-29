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
