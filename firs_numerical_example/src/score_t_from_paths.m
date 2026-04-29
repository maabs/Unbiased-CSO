function S = score_t_from_paths(y, X_paths, init_pars, trans_pars, obs_pars)
%SCORE_T_FROM_PATHS  Full score for t-observation SSM
%
% Computes:
%   - init score
%   - transition score
%   - observation score (t-student, wrt log sigma)
%
% Inputs:
%   y         : 1×T
%   X_paths   : 1×T×P
%   init_pars : parameters of initial state density
%   trans_pars: parameters of Gaussian transition
%   obs_pars  : parameters of t-Student obs: (v, sigma)
%
% Output struct S with fields:
%   .per_path_init
%   .per_path_trans   (vector or matrix depending on parameters)
%   .per_path_obs
%   .avg_init
%   .avg_trans
%   .avg_obs

    [~, T, P] = size(X_paths);

    % ----- 1) INITIAL SCORE (Gaussian) -----
    % grad_log_p1 should return gradient wrt initial parameters
    S_init = zeros(1, P);
    for p = 1:P
        x1 = X_paths(1,1,p);
        S_init(p) = grad_log_p1_gauss(x1, init_pars);
    end

    % ----- 2) TRANSITION SCORE (Gaussian AR(1)) -----
    % grad_log_f(x_t, x_{t-1}, trans_pars)
    S_trans = zeros(2, P);
    for p = 1:P
        x_p = squeeze(X_paths(1,:,p));  % 1×T → row vector
        s = 0;
        for t = 2:T
            s = s + grad_log_f_gauss(x_p(t), x_p(t-1), trans_pars);
        end
        S_trans(:,p) = s;
    end

    % ----- 3) OBSERVATION SCORE (t-Student) -----
    S_obs = grad_log_g_t_stud_vec(y, X_paths, obs_pars);   % 1×P

    % ----- 4) Build output -----
    S.per_path_init  = S_init;
    S.per_path_trans = S_trans;
    S.per_path_obs   = S_obs;

    S.avg_init  = mean(S_init);
    S.avg_trans = mean(S_trans,2);
    S.avg_obs   = mean(S_obs);
end


