function ll = gaussLogScore(y, Sigma)
%GAUSSLOGSCORE Log N(y;0,Sigma) while ignoring NaNs in y via subsetting.
% Robust to non-PD Sigma by adaptive diagonal jitter.

    % Ensure column vector
    y = y(:);

    % Keep only observed entries (finite is stricter than ~isnan)
    ind = isfinite(y);
    y = y(ind);

    % If nothing observed, likelihood is undefined; choose NaN or 0.
    % (NaN is safer so you notice it; use 0 if you prefer "no contribution".)
    n = numel(y);
    if n == 0
        ll = NaN;
        return;
    end

    % Subset covariance
    S = Sigma(ind, ind);

    % Guard against NaN/Inf in covariance (common culprit in EWMA streams)
    if ~all(isfinite(S(:)))
        ll = NaN;
        return;
    end

    % Enforce symmetry (numerical drift)
    S = (S + S')/2;

    % Attempt Cholesky with adaptive jitter
    [C, p] = chol(S, 'lower');
    if p ~= 0
        % Scale jitter to matrix magnitude (important!)
        base = max(trace(S)/n, 1);        % avoid 0 scaling
        jitter = 1e-12 * base;

        for k = 1:12
            [C, p] = chol(S + jitter*eye(n), 'lower');
            if p == 0
                break;
            end
            jitter = jitter * 10;
        end

        if p ~= 0
            % Still not PD -> give NaN (or -Inf if you want to penalize hard)
            ll = NaN;
            return;
        end
    end

    % Quadratic form and logdet via Cholesky (stable)
    alpha  = C \ y;
    quad   = alpha' * alpha;
    logdet = 2 * sum(log(diag(C)));

    ll = -0.5*( n*log(2*pi) + logdet + quad );
end