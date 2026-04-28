function Sigmar_beta_hat = mc_estimate_Sigmar_beta(LW, lambdas, omegas, sigma2, Givset, beta, M)
% Unbiased estimator of Sigmar*beta without building Sigmar

[N,~] = size(LW);
beta = beta(:);

if isscalar(sigma2), sig = sqrt(sigma2)*ones(N,1);
else,               sig = sqrt(sigma2(:));
end

% Store r samples only through running mean and second moment action
% We'll use the identity: \hat{Cov} * beta = (1/(M-1)) * sum (r-mu)(r-mu)'beta
% Implemented via Welford-like updates maintaining Cbeta = sum delta * (delta2' * beta)

mu = zeros(N,1);
Cbeta = zeros(N,1);

for m = 1:M
    u = sample_factor_u(lambdas, omegas, Givset);
    y = LW*u + sig.*randn(N,1);
    r = exp(y);

    if m == 1
        mu = r;
    else
        delta  = r - mu;
        mu     = mu + delta/m;
        delta2 = r - mu;

        % rank-1 update to covariance-action:
        % C += delta * delta2'  =>  C*beta += delta * (delta2' * beta)
        Cbeta = Cbeta + delta * (delta2' * beta);
    end
end

Sigmar_beta_hat = Cbeta / (M-1);
end


function u = sample_factor_u(lambdas, omegas, Givset)
% u ~ N(0, Sigma_f) where Sigma_f = G*diag(lambdas)*G'
    K = numel(lambdas);
    z = randn(K,1);
    v = sqrt(lambdas(:)) .* z;                  % diag(sqrt(lam))*z
    u = apply_givens_vec(v, omegas, Givset, 'G'); % u = G*v
end
function Sigmar_hat = mc_estimate_Sigmar(LW, lambdas, omegas, sigma2, Givset, M)
% Unbiased estimator of Sigmar = Cov(exp(y)), y ~ N(0, Sigmay)
% Sigmay = diag(sigma2) + LW*Sigma_f*LW', Sigma_f = G*diag(lambdas)*G'

[N,K] = size(LW);

% sigma2 handling
if isscalar(sigma2), sig = sqrt(sigma2)*ones(N,1);
else,               sig = sqrt(sigma2(:));
end

% Welford accumulator for covariance (numerically stable)
mu  = zeros(N,1);
C   = zeros(N,N);

for m = 1:M
    u = sample_factor_u(lambdas, omegas, Givset);  % Kx1
    y = LW*u + sig.*randn(N,1);                    % Nx1
    r = exp(y);                                    % gross returns

    if m == 1
        mu = r;
    else
        delta  = r - mu;
        mu     = mu + delta/m;
        delta2 = r - mu;
        C      = C + delta*delta2.';               % rank-1 update
    end
end

Sigmar_hat = C / (M-1);
end