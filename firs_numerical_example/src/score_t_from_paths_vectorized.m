function S = score_t_from_paths_vectorized(y, X_paths, init_pars, trans_pars, obs_pars)
%SCORE_T_FROM_PATHS_VECTORIZED
%   Full score for t-observation SSM using *vectorized* init and trans grads.
%
% Inputs:
%   y         : 1×T                (observations)
%   X_paths   : 1×T×P              (P trajectories, e.g. from PG)
%   init_pars : struct, parameters of initial distribution
%   trans_pars: struct, parameters of Gaussian transition
%   obs_pars  : struct, parameters of t-Student obs (e.g. v, sigma)
%
% Required helper functions (vectorized):
%   G_init  = grad_log_p1_gauss_vec(X1, init_pars);
%            % X1: 1×P, G_init: d_init×P  (or 1×P if scalar)
%
%   G_trans = grad_log_f_gauss_vec(X_paths, trans_pars);
%            % X_paths: 1×T×P, G_trans: d_trans×P
%
%   g_obs   = grad_log_g_t_stud_vec(y, X_paths, obs_pars);
%            % y: 1×T, X_paths: 1×T×P, g_obs: 1×P (wrt log sigma)
%
% Output:
%   S : struct with fields
%       .per_path_init   : d_init×P   (or 1×P)
%       .per_path_trans  : d_trans×P
%       .per_path_obs    : 1×P
%       .avg_init        : d_init×1   (mean over paths)
%       .avg_trans       : d_trans×1
%       .avg_obs         : scalar

    [~, T, P] = size(X_paths); %#ok<ASGLU>

    % 1) INITIAL SCORE (vectorized)
    % Extract x_1 for each path: X_paths(1,1,p), p=1..P
    X1 = squeeze(X_paths(1,1,:)).';   % 1×P

    G_init = grad_log_p1_gauss_vec(X1, init_pars);  % d_init×P or 1×P

    % 2) TRANSITION SCORE (vectorized)
    % grad_log_f_gauss_vec handles all t>=2 and all paths internally
    G_trans = grad_log_f_gauss_vec(X_paths, trans_pars);  % d_trans×PxT or d_transx(T-1)xP

    % 3) OBSERVATION SCORE (t-Student, vectorized)
    % This should already be vectorized over t and p
    g_obs = grad_log_g_t_stud_vec(y, X_paths, obs_pars);  % (1×P), notice that
    % the sum wrt to the time is already computed.

    % 4) Pack into output struct
    S.per_path_init   = G_init;          % d_init×P
    S.per_path_trans  = G_trans;         % d_trans×x(T-1)xP
    S.per_path_obs    = g_obs;           % 1×P

    S.avg_init  = mean(G_init,  2);      % d_init×1
    S.avg_trans = squeeze(mean(sum(G_trans, 2),3));      % d_trans×1
    S.avg_obs   = mean(g_obs,   1);      % scalar

end

