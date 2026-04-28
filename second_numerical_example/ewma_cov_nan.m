function Sigmas = ewma_cov_nan(Y, alpha, H)
% Y: N x Ttrain, NaNs allowed
% alpha: smoothing parameter (e.g., 0.06 or 2/(1+span))
% H: number of forecast steps; we output the last EWMA repeated H times
%
% Output: Sigmas is N x N x H, all equal to final EWMA (random walk forecast)

[N,T] = size(Y);

% Initialize with pairwise covariance (stable start)
Sigma = cov_pairwise_nan(Y);

for t=1:T
    yt = Y(:,t);
    ind = ~isnan(yt);
    if any(ind)
        r = yt(ind);
        outer = r*r';              % observed block
        Sigma(ind,ind) = (1-alpha)*Sigma(ind,ind) + alpha*outer;
        % For blocks involving missing entries: leave unchanged
    end
    Sigma = (Sigma+Sigma')/2;
    Sigma = Sigma + 1e-10*eye(N);
end

Sigmas = repmat(Sigma, 1, 1, H);    % multi-step EWMA forecast is constant absent dynamics
end