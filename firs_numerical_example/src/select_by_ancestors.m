function X_post = select_by_ancestors(X, A)
%SELECT_BY_ANCESTORS  Gather columns per filter using ancestor indices.
%   X_post = select_by_ancestors(X, A)
%   X : d_x × N × M       (particles, pre-resampling)
%   A : N × M (uint32)    (ancestor indices in 1..N per filter)
%   X_post : d_x × N × M  (post-resample cloud)
%
% Flip the mode below to 'vectorized' to use the no-loop gather.

    % ===== choose implementation here =====
    mode = "loop";    % "loop" or "vectorized"
    % ======================================

    % ---- basic checks ----
    if ndims(X) ~= 3
        error('X must be 3-D (d_x × N × M).');
    end
    if ~ismatrix(A)
        error('A must be 2-D (N × M).');
    end

    [d_x, N, M] = size(X);
    [Na, Ma]    = size(A);
    if Na ~= N || Ma ~= M
        error('Size mismatch: X is d_x×N×M = %d×%d×%d, but A is %d×%d.', d_x, N, M, Na, Ma);
    end
    if any(A(:) < 1 | A(:) > N)
        error('Ancestor indices in A must be within 1..N.');
    end

    switch mode
        case "loop"
            % Clear, parfor-ready per-filter gather (often fastest for small/medium M)
            X_post = zeros(d_x, N, M, 'like', X);
            for m = 1:M
                idx = double(A(:, m));        % 1..N indices for filter m
                X_post(:, :, m) = X(:, idx, m);
            end

        case "vectorized"
            % Vectorized gather using a single indexed take over stacked pages
            % 1) reshape pages side-by-side: d_x × (N*M)
            X23  = reshape(X, d_x, N*M);

            % 2) convert per-page indices to global 1..N*M via block offsets
            %    offs = [0, N, 2N, ..., (M-1)N]
            offs = (0:M-1) * N;                      % 1×M
            % Use bsxfun for broad MATLAB compatibility (instead of implicit expansion)
            J = bsxfun(@plus, double(A), offs);      % N×M

            % 3) gather, then reshape back to d_x × N × M
            X_post = reshape(X23(:, J(:)), d_x, N, M);

        otherwise
            error('Unknown mode "%s". Use "loop" or "vectorized".', mode);
    end
end
