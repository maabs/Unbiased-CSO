function out = score_gaussian_from_paths(y, X_paths, theta, q, r, S0, exclude_first_col)
% y: 1×T, X_paths: 1×T×M×B1
if nargin < 7, exclude_first_col = false; end
obs_pars   = struct('R', r);
trans_pars = struct('theta', theta, 'q', q);
init_pars  = struct('S0', S0);

[M, B1] = deal(size(X_paths,3), size(X_paths,4));
paths_sel = true(M,B1);
if exclude_first_col, paths_sel(:,1) = false; end

out = score_from_paths(y, X_paths, ...
        @grad_log_g_gauss, @grad_log_f_gauss, @grad_log_p1_gauss, ...
        obs_pars, trans_pars, init_pars, paths_sel, []);
end


%% ===== helper: one run =====
