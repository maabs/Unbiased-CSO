function ll = g_mix_t_gauss(yt, Xt, pars, t)
%G_MIX_T_GAUSS  Log-likelihood for a mixture of t-Student and Gaussian.
%
%   ll = g_mix_t_gauss(yt, Xt, pars, t)
%
% Inputs:
%   yt   : 1×1   (or 1×d_y, d_y=1) observation at time t
%   Xt   : 1×N×M particles at time t
%   pars : struct with fields
%          .R      : Gaussian observation variance r
%          .v      : t-Student degrees of freedom nu
%          .sigma  : t-Student scale sigma
%          .m_mix  : mixing parameter m (scalar)
%                    weight_t   = 1/(m_mix+1)
%                    weight_gauss = m_mix/(m_mix+1)
%   t    : time index (unused here, but kept for interface consistency)
%
% Output:
%   ll   : 1×N×M log-likelihood of the mixture

    R     = pars.R;
    v     = pars.v;
    sigma = pars.sigma;
    m_mix = pars.m_mix;
    
        % ---- Check whether sigma^2 and R are (numerically) equal ----
    tol = 1e-12;
    same_var = abs(sigma^2 - R) <= tol * max(1, abs(R));

    if same_var
        % Optional: display a notice for debugging once
         persistent warned
         if isempty(warned)
             warning('g\_mix\_t\_gauss: sigma^2 and R are numerically equal.');
             warned = true;
         end
    end

    % ----- residuals -----
    % Xt is 1×N×M, yt is 1×1 ⇒ broadcast to 1×N×M
    diff = yt - Xt;      % 1×N×M

    % ----- Gaussian log-density -----
    % N(Xt, R)
    ll_gauss = -0.5 * (diff.^2) / R ...
               - 0.5 * log(2*pi*R);     % 1×N×M

    % ----- t-Student log-density (1D) -----
    % f(y|x) = c * ( 1 + (diff^2)/(v*sigma^2) )^{-(v+1)/2}
    z = (diff ./ sigma).^2;             % 1×N×M

    const_t = gammaln((v+1)/2) ...
              - gammaln(v/2) ...
              - 0.5*log(v*pi) ...
              - log(sigma);

    ll_t = const_t ...
           - 0.5*(v+1) .* log(1 + z./v);   % 1×N×M

    % ----- mixture weights -----
    w_t = 1/(m_mix + 1);
    w_g = m_mix/(m_mix + 1);

    log_w_t = log(w_t);
    log_w_g = log(w_g);

    A = ll_t     + log_w_t;   % 1×N×M
    B = ll_gauss + log_w_g;   % 1×N×M

    % ----- log-sum-exp for mixture -----
    Lmax = max(A, B);                         % 1×N×M
    ll = Lmax + log( exp(A - Lmax) + exp(B - Lmax) );  % 1×N×M
end



