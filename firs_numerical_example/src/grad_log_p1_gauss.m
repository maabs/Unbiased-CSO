function g = grad_log_p1_gauss(x1, init_pars)
% init_pars.S0
S0 = init_pars.S0;
g  = -0.5/S0 + 0.5*(x1.^2)/(S0^2);  % scalar
end

