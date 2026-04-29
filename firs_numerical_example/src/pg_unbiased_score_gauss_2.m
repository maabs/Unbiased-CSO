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


