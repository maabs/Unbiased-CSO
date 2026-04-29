function Gf = grad_log_f_gauss_vec(Xp, trans_pars)
% Xp : 1×T×P
% Gf : 2×(T-1)×P  (row 1: dtheta, row 2: dq)
theta = trans_pars.theta;
q     = trans_pars.q;
x_tm1 = Xp(:, 1:end-1, :);                   % 1×(T-1)×P
x_t   = Xp(:, 2:end,   :);                   % 1×(T-1)×P
res   = x_t - theta .* x_tm1;                % 1×(T-1)×P

dtheta = (res .* x_tm1) ./ q;                % 1×(T-1)×P
dq     = -0.5./q + 0.5*(res.^2)./(q.^2);     % 1×(T-1)×P

Gf = cat(1, dtheta, dq);                      % 2×(T-1)×P
end

% Initial gradient (wrt S0): returns p_init×P with p_init=1
