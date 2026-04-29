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


