function out = pf_parallel( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    seed)
%pf_parallel_LOGW  Bootstrap PF (M i.i.d. filters) storing log-weights.
%
% out = pf_parallel_logw(y, T, N, M, in_dist_samp, in_pars, ...
%     trans_dist_samp, trans_pars, g, g_pars, seed)
%
% Inputs are identical to pf_parallel; this version additionally returns:
%   out.logw : N × T × M   (normalized log-weights at each time, pre-resampling)
%
% Notes:
% - Resamples EVERY step (independently per filter).
% - Uses log-sum-exp numerics; if all log-lik = -Inf for some filter/time,
%   sets weights to uniform so logw = -log(N).

if nargin < 11 || isempty(seed), seed = 12345; end

% --- basic checks ---
[~, T_chk] = size(y);
if T_chk ~= T
    error('size(y,2)=%d does not match T=%d.', T_chk, T);
end

% --- RNG: one stream per filter ---
streams = cell(1, M);
for m = 1:M
    streams{m} = RandStream('mrg32k3a', 'Seed', seed + m - 1);
end

% --- initialize particles at t=1 (pre-resampling) ---
X = in_dist_samp(in_pars, N, M);         % d_x × N × M
[d_x, N_chk, M_chk] = size(X);
if N_chk ~= N || M_chk ~= M
    error('in_dist_samp returned [%d %d %d], expected d_x×N×M with N=%d, M=%d.', ...
          d_x, N_chk, M_chk, N, M);
end

% --- preallocate outputs ---
particles = zeros(d_x, N, T, M);
ancestors = zeros(N, T, M, 'uint32');
logw_hist = -inf(N, T, M);   % normalized log-weights

% ===================== main loop =====================
for t = 1:T
    % store pre-resampling cloud
    particles(:, :, t, :) = X;

    % log-likelihoods: ll is 1×N×M
    ll = g(y(:, t), X, g_pars, t);
    sz = size(ll);
    if numel(sz) ~= 3 || sz(1) ~= 1 || sz(2) ~= N || sz(3) ~= M
        error('g must return 1×N×M; got [%s].', strjoin(string(sz), '×'));
    end

    % ---- normalize log-weights -> w_norm (N×M), with all-Inf guard ----
    L = squeeze(permute(ll, [2 3 1]));        % N × M  (log unnormalized weights)
    Lmax    = max(L, [], 1);
    shifted = L - Lmax;                        % log-sum-exp trick
    W       = exp(shifted);                    % N × M (unnormalized)
    S       = sum(W, 1);                       % 1 × M
    allInf  = (S == 0);

    % normalized weights
    w_norm = zeros(N, M);
    if any(~allInf), w_norm(:, ~allInf) = W(:, ~allInf) ./ S(1, ~allInf); end
    if any(allInf),  w_norm(:,  allInf) = 1 / N;                           end

    % ---- store normalized LOG-weights: logw = L - logsumexp(L) ----
    logw = -inf(N, M);
    if any(~allInf)
        denom = Lmax(1, ~allInf) + log(S(1, ~allInf));   % 1 × (#good columns)
        % subtract denom from each row in those columns
        logw(:, ~allInf) = bsxfun(@minus, L(:, ~allInf), denom);
    end
    if any(allInf)
        logw(:, allInf) = -log(N);   % uniform fallback → log(1/N)
    end
    % place into N × T × M at time t
    logw_hist(:, t, :) = reshape(logw, N, 1, M);

    % ---- multinomial resampling per filter (sorted-uniform scan) ----
    A = multinomial_resample_sorted(w_norm, streams);  % N×M (uint32)
    ancestors(:, t, :) = A;

    % ---- form post-resample cloud and propagate (except after final t) ----
    if t < T
        X_post = select_by_ancestors(X, A);            % d_x×N×M
        X = trans_dist_samp(X_post, trans_pars, t+1);  % d_x×N×M
        if ~isequal(size(X), [d_x, N, M])
            error('trans_dist_samp returned [%s], expected d_x×N×M.', ...
                  strjoin(string(size(X)), '×'));
        end
    end
end
% =====================================================

% package outputs
out.particles = particles;
out.ancestors = ancestors;
out.logw      = logw_hist;   % N × T × M

end % pf_parallel_logw

% ---------- helpers ----------

