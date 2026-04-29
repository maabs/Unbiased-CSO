function g = grad_log_g_gauss(y_t, x_t, obs_pars, ~)
% y_t, x_t are scalars here (1×1), obs_pars.R = r
r = obs_pars.R;
res2 = (y_t - x_t).^2;
g = -0.5/r + 0.5*res2/(r^2);   % scalar gradient wrt r
end

