function ll = msv_logscore_caseA(y, B, G, lambdas, sigma2)
% ll = log N(y; 0,  B*G*diag(lambdas)*G'*B' + sigma2*I )
%
% y      : N x 1 (can contain NaN)
% B      : N x K  (this is LW = L.*Weights)
% G      : K x K  orthogonal (product of givens)
% lambdas: K x 1  positive (typically exp(hNew))
% sigma2 : scalar > 0

% ---- handle missing data by subsetting ----
ind = ~isnan(y);
y = y(ind);
B = B(ind,:);

n = length(y);
K = size(B,2);

% Rotate loadings: U = B*G
U = B * G;                      % n x K

% Build A = Lambda^{-1} + (1/sigma2) U'U   (K x K)
invLam = diag(1./lambdas(:));   % K x K (lambdas positive)
UtU = U' * U;                   % K x K
A = invLam + (1/sigma2) * UtU;

% Cholesky of A (KxK)
% add tiny jitter if needed
jitter = 1e-12;
[R,p] = chol(A + jitter*eye(K));   % A = R'R with MATLAB chol default upper
if p > 0
    % fallback with a bit more jitter
    jitter = 1e-8;
    [R,p] = chol(A + jitter*eye(K));
    if p > 0
        error('A is not SPD even after jitter. Check lambdas/sigma2.');
    end
end

% b = (1/sigma2) U' y
b = (1/sigma2) * (U' * y);      % K x 1

% Solve x = A^{-1} b via Cholesky: A = R'R
x = R \ (R' \ b);

% Quadratic form: y' Sigma^{-1} y
quad = (1/sigma2) * (y' * y) - (b' * x);

% Log-determinant: log|Sigma| = n log(sigma2) + log|Lambda| + log|A|
logdetA = 2*sum(log(diag(R)));
logdetLam = sum(log(lambdas(:)));
logdetSigma = n*log(sigma2) + logdetLam + logdetA;

% Log-likelihood
ll = -0.5*( n*log(2*pi) + logdetSigma + quad );
end