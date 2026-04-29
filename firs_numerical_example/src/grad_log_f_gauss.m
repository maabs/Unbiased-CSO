function g = grad_log_f_gauss(x_t, x_tm1, trans_pars, ~)
% trans_pars: .theta, .q
theta = trans_pars.theta; q = trans_pars.q;
res   = x_t - theta*x_tm1;
dtheta = (res * x_tm1) / q;
dq     = -0.5/q + 0.5*(res.^2)/(q^2);
g = [dtheta; dq];               % 2×1
end

