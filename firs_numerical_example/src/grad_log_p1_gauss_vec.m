function G0 = grad_log_p1_gauss_vec(X1, init_pars)
% X1 : 1×1×P   (initial state per path)
% G0 : 1×P
S0 = init_pars.S0;
x1 = squeeze(X1);                             % 1×P
G0 = -0.5./S0 + 0.5*(x1.^2)./(S0.^2);        % 1×P
end

%===== scoring helpers (same as earlier) =====

