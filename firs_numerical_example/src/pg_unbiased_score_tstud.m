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


