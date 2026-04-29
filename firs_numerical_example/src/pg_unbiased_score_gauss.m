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

