function [Sigmar_beta, mu_r] = Sigmar_times_beta_exact(beta, LW, omegas, lambdas, sigma2, Givset, blockSize)
% EXACT compute Sigmar * beta where y ~ N(0, Sigmay), r = exp(y),
% Sigmay = diag(sigma2) + LW * (G*diag(lambdas)*G') * LW'
%
% Inputs:
%   beta      : N x 1
%   LW        : N x K
%   omegas    : (#rot) x 1   (for Givens)
%   lambdas   : K x 1        (positive)
%   sigma2    : scalar or N x 1 (obs noise variances)
%   Givset    : (#rot) x 2
%   blockSize : e.g. 512 or 1024
%
% Output:
%   Sigmar_beta : N x 1 exact

if nargin < 7 || isempty(blockSize), blockSize = 512; end

[N,K] = size(LW);
beta = beta(:);

% ---- Build B = LW * G * diag(sqrt(lambdas)) without forming G explicitly ----
% Step A: U = LW' (K x N), apply G' or G appropriately to columns.
% We need LW*G, so we can compute (LW*G) = (G'*LW')' .
U = LW';                               % K x N
U = apply_givens_mat(U, omegas, Givset, 'GT');  % apply G' on the left to U (row rotations)
LG = U';                                % N x K equals LW*G
B  = LG .* (sqrt(lambdas(:))');         % N x K scale columns by sqrt(lambdas)

% ---- Diagonal of Sigmay ----
if isscalar(sigma2)
    d = sigma2 * ones(N,1);
else
    d = sigma2(:);
end
diagY = d + sum(B.^2, 2);               % N x 1
m = exp(0.5 * diagY);                   % mean of exp(y)
mu_r=m- ones(size(m));
% ---- v = m .* beta and its sum ----
v = m .* beta;
vsum = sum(v);

Sigmar_beta = zeros(N,1);

% ---- Stream blocks of rows ----
for i0 = 1:blockSize:N
    i1 = min(N, i0 + blockSize - 1);
    I = i0:i1;

    % Z = B(I,:) * B' gives b x N matrix of off-diagonal dot products
    Z = B(I,:) * B.';                   % (|I|) x N

    % Fix diagonal entries to include the D term exactly:
    % For row corresponding to global index ii in I, column ii should be diagY(ii)
    % Currently Z(local, ii) = b_ii' b_ii = sum(B(ii,:).^2)
    % So add d(ii) to that entry.
    for t = 1:numel(I)
        ii = I(t);
        Z(t, ii) = Z(t, ii) + d(ii);
    end

    % Elementwise exp
    E = exp(Z);

    % s_I = sum_j exp(Sigmay_ij) * v_j
    s = E * v;

    % Sigmar*beta for rows I: m_i * ( s_i - sum_j v_j )
    Sigmar_beta(I) = m(I) .* (s - vsum);

end
end


function U = apply_givens_mat(U, omegas, Givset, mode)
% Apply G or G' to a KxN matrix by rotating pairs of rows.
% This is the matrix analogue of your apply_givens_vec, but acting on all columns.
    c = cos(omegas(:));
    s = sin(omegas(:));
    R = size(Givset,1);

    if strcmp(mode,'G')
        for k = 1:R
            i = Givset(k,1); j = Givset(k,2);
            Ui = U(i,:); Uj = U(j,:);
            U(i,:) = c(k)*Ui - s(k)*Uj;
            U(j,:) = s(k)*Ui + c(k)*Uj;
        end
    elseif strcmp(mode,'GT')
        for k = R:-1:1
            i = Givset(k,1); j = Givset(k,2);
            Ui = U(i,:); Uj = U(j,:);
            U(i,:) = c(k)*Ui + s(k)*Uj;
            U(j,:) = -s(k)*Ui + c(k)*Uj;
        end
    else
        error('mode must be G or GT');
    end
end