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


