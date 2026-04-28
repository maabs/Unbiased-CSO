%% CPF vs CPF_PARALLEL timing header (for Ibex batch test)
% This header assumes:
%  - This .m file also contains the function definitions for:
%       cpf, cpf_parallel, multinomial_resample_sorted, select_by_ancestors, draw_from_logw_np, etc.
%  - You submit a SLURM job with --ntasks-per-node set (e.g., 8)
%  - OMP_NUM_THREADS is set to 1 in the .sbatch script

clear; clc;

fprintf('=== CPF vs CPF_PARALLEL timing test ===\n');

%% 1) Detect number of cores from SLURM and start parpool for cpf_parallel
cores = str2double(getenv('SLURM_CPUS_ON_NODE'));
if isnan(cores) || cores <= 0
    cores = 1;
end
fprintf('SLURM_CPUS_ON_NODE = %d\n', cores);

pool = [];
if cores > 1
    try
        pc = parcluster('local');
        pc.JobStorageLocation = fullfile(pwd, 'folders');  % or absolute path
        if ~exist(pc.JobStorageLocation, 'dir')
        mkdir(pc.JobStorageLocation);
        end

        pool = parpool(pc, cores);
        fprintf('Started parpool with %d workers for cpf_parallel.\n', cores);
    catch ME
        fprintf('Could not start parpool, falling back to serial only.\n');
        disp(ME.message);
        pool = [];
        cores = 1;
    end
else
    fprintf('Running in pure serial mode (cores = 1).\n');
end

%% 2) Model setup: 1D linear Gaussian state-space model (LGSSM)
T          = 10;       % time steps
theta_true = 0.95;      % AR(1) coefficient
q_true     = 0.2^2;     % state noise variance
r_true     = 0.3^2;     % obs noise variance
S0_true    = q_true/(1 - theta_true^2);   % stationary initial variance

rng(42);   % reproducible
x_true = zeros(1, T);
x_true(1) = sqrt(S0_true) * randn;
for t = 2:T
    x_true(t) = theta_true * x_true(t-1) + sqrt(q_true)*randn;
end
y = x_true + sqrt(r_true)*randn(1, T);    % observations (H = 1)

[d_x, d_y] = deal(1, 1);  %#ok<NASGU>

%% 3) User functions: in_dist_samp, trans_dist_samp, g, trans_logpdf

% Initial distribution: N(0, S0_true)
in_pars.mu    = 0;
in_pars.Sigma = S0_true;
in_dist_samp  = @(p, N, M) reshape(p.mu + sqrt(p.Sigma) * randn(1, N*M), 1, N, M);

% Transition: X_t = theta * X_{t-1} + sqrt(q)*eps
tr_pars.theta = theta_true;
tr_pars.q     = q_true;
tr_pars.sig   = sqrt(q_true);
trans_dist_samp = @(Xprev, p, t) ...
    p.theta * Xprev + p.sig * randn(size(Xprev), 'like', Xprev);

% Observation log-likelihood: y_t | x_t ~ N(x_t, r_true)
g_pars.R = r_true;
g = @(yt, Xt, p, t) ...
    reshape( -0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R), ...
             1, size(Xt,2), size(Xt,3));

% Transition log-density: for backward simulation or CPF tests
trans_logpdf = @(x_next, X_prev, pars, t) ...
    -0.5*((x_next - theta_true*X_prev).^2)/q_true - 0.5*log(2*pi*q_true);  % 1×N

%% 4) CPF parameters: particles N, filters M, reference path x_ref
N      = 10;                 % particles per filter
M      = 50000*max(cores, 2);       % number of parallel filters; at least as large as core count
seed0  = 12345;               % base RNG seed for CPF/CPF_PARALLEL
traj_mode = "backward";       % or "ancestors"

% Reference path: use the true latent state (size 1×T×M)
x_ref = reshape(x_true, d_x, T, 1);   % 1 × T × 1
x_ref = repmat(x_ref, 1, 1, M);       % 1 × T × M

% Flags for optional outputs (to keep memory light in timing tests)
store_particles = false;
store_ancestors = false;
store_logw      = false;

%% 5) Timing: compare cpf (serial) vs cpf_parallel (parallel over M)

NRuns = 5;   % do multiple runs and average to smooth out noise

fprintf('\n=== Timing CPF (serial) ===\n');
t_serial = 0;
for r = 1:NRuns
    tic;
    out_cpf = cpf( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, tr_pars, ...
        g, g_pars, ...
        x_ref, seed0 + r, ...
        traj_mode, trans_logpdf, ...
        store_particles, store_ancestors, store_logw);
    t_serial = t_serial + toc;
end
t_serial = t_serial / NRuns;
fprintf('Average CPF time over %d runs: %.3f seconds\n', NRuns, t_serial);

fprintf('\n=== Timing CPF_PARALLEL (parfor over M) ===\n');
t_parallel = 0;
for r = 1:NRuns
    tic;
    out_cpfp = cpf_parallel( ...
        y, T, N, M, ...
        in_dist_samp, in_pars, ...
        trans_dist_samp, tr_pars, ...
        g, g_pars, ...
        x_ref, seed0 + r, ...
        traj_mode, trans_logpdf, ...
        store_particles, store_ancestors, store_logw);
    t_parallel = t_parallel + toc;
end
t_parallel = t_parallel / NRuns;
fprintf('Average CPF_PARALLEL time over %d runs: %.3f seconds\n', NRuns, t_parallel);

speedup = t_serial / t_parallel;
fprintf('\n=== Summary ===\n');
fprintf('Cores (SLURM_CPUS_ON_NODE): %d\n', cores);
fprintf('Filters M:                  %d\n', M);
fprintf('Average CPF time:           %.3f s\n', t_serial);
fprintf('Average CPF_PARALLEL time:  %.3f s\n', t_parallel);
fprintf('Speedup (serial / parallel) = %.2f\n', speedup);

% Optionally, clean up the pool at the end of the batch job
if ~isempty(pool)
    delete(pool);
end

fprintf('=== End of CPF timing header ===\n');
%% -------- local function: 1D Kalman filter --------
function [m_hist, P_hist] = kalman_1d(y, rho, q, r, H, m0, P0)
% y   : 1 × T
% rho : scalar state coefficient
% q   : process variance
% r   : observation variance
% H   : scalar observation matrix
% m0, P0 : prior mean/var for x_1
    T = size(y, 2);
    m = m0; P = P0;
    m_hist = zeros(1, T);
    P_hist = zeros(1, T);

    for t = 1:T
        % --- update with y_t ---
        S     = H*P*H' + r;              % innovation variance
        K     = (P*H') / S;              % Kalman gain
        innov = y(1, t) - H*m;           % innovation
        m     = m + K*innov;             % posterior mean
        P     = (1 - K*H)*P;             % posterior variance

        m_hist(t) = m;
        P_hist(t) = P;

        % --- predict to t+1 (skip after final step) ---
        if t < T
            m = rho*m;
            P = rho*P*rho + q;           % since scalar, rho' = rho
        end
    end
end

%%%
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

function A = multinomial_resample_sorted(w_norm, streams)
% w_norm: N×M (columns sum to 1)
% streams: {1×M} RandStream
    [N, M] = size(w_norm);
    A = zeros(N, M, 'uint32');
    for m = 1:M
        u   = sort(rand(streams{m}, N, 1));   % sorted uniforms
        cdf = cumsum(w_norm(:, m));           % cumulative weights
        i = 1; j = 1;
        while i <= N
            while u(i) > cdf(j)               % advance CDF pointer
                j = j + 1;
            end
            A(i, m) = uint32(j);
            i = i + 1;
        end
    end
end


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
function out = cpf( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    x_ref, seed, ...
    traj_mode, trans_logpdf, ...
    store_particles, store_ancestors, store_logw)
%CPF_PARALLEL  Conditional bootstrap PF with selectable trajectory sampler.
%
% out = cpf_parallel(y, T, N, M, in_dist_samp, in_pars, trans_dist_samp, trans_pars, ...
%                    g, g_pars, x_ref, seed, traj_mode, trans_logpdf, ...
%                    store_particles, store_ancestors, store_logw)
%
% Required:
%   y                : d_y × T                  (shared across filters)
%   T, N, M          : time steps, #particles, #filters
%   in_dist_samp     : @(in_pars,N,M) -> d_x×N×M  initial particles
%   trans_dist_samp  : @(X_prev,trans_pars,t) -> d_x×N×M  transition sampler
%   g                : @(y_t,X_t,g_pars,t) -> 1×N×M       LOG-likelihoods
%   x_ref            : d_x × T × M             (reference paths, one per filter)
%
% Optional (defaults in code):
%   seed             : scalar RNG seed (default 12345)
%   traj_mode        : "ancestors" (default) or "backward"
%   trans_logpdf     : required iff traj_mode=="backward"
%                      @(x_next, X_prev, trans_pars, t) -> 1×N  LOG transition density
%   store_particles  : (logical) return out.particles (default true)
%   store_ancestors  : (logical) return out.ancestors (default true)
%   store_logw       : (logical) return out.logw      (default true)
%
% Outputs (always):
%   out.sampled_path : d_x × T × M    (sampled trajectory per filter)
%   out.sampled_idx  : 1 × M          (terminal index at time T)
%   out.sampled_mode : "ancestors" or "backward"
%
% Outputs (optional; see store_* flags):
%   out.particles    : d_x × N × T × M   (pre-resampling clouds)
%   out.ancestors    : N × T × M (uint32)
%   out.logw         : N × T × M   (normalized log-weights, pre-resampling)
%
% Notes:
% - Conditional path is fixed in slot j=1 across time (A(1,:)=1; X(:,1,t)=x_ref(:,t)).
% - Resampling is multinomial, independently per filter, every time step.
% - Internally, the function keeps the necessary arrays to produce the trajectory.
%   The store_* flags only control what is returned in 'out'.

% ----------------- defaults & input handling -----------------
if nargin < 12 || isempty(seed),       seed = 12345;        end
if nargin < 13 || isempty(traj_mode),  traj_mode = "ancestors"; end
traj_mode = lower(string(traj_mode));

if nargin < 14, trans_logpdf = []; end
if traj_mode == "backward" && ~isa(trans_logpdf,'function_handle')
    error('traj_mode="backward" requires a valid trans_logpdf function handle.');
end

if nargin < 15 || isempty(store_particles),  store_particles  = true; end
if nargin < 16 || isempty(store_ancestors),  store_ancestors  = true; end
if nargin < 17 || isempty(store_logw),       store_logw       = true; end

% ----------------- basic checks -----------------
[dy, T_chk] = size(y); %#ok<NASGU>
if T_chk ~= T, error('size(y,2)=%d does not match T=%d.', T_chk, T); end

% --- RNG: one stream per filter for clean independence ---
streams = cell(1, M);
parfor m = 1:M
    streams{m} = RandStream('mrg32k3a', 'Seed', seed + m - 1);
end

% --- initialize particles at t=1 (pre-resampling) ---
X = in_dist_samp(in_pars, N, M);           % d_x × N × M
[d_x, N_chk, M_chk] = size(X);
if N_chk ~= N || M_chk ~= M
    error('in_dist_samp returned [%d %d %d], expected d_x×N×M with N=%d, M=%d.', ...
          d_x, N_chk, M_chk, N, M);
end

% --- accept x_ref as d_x×T or d_x×T×1 and expand to d_x×T×M ---
if ndims(x_ref) == 2
    if ~isequal(size(x_ref), [d_x, T])
        error('x_ref must be d_x×T (or d_x×T×M). Got %s.', mat2str(size(x_ref)));
    end
    x_ref = reshape(x_ref, d_x, T, 1);
end
if size(x_ref,3) == 1 && M > 1
    x_ref = repmat(x_ref, 1, 1, M);
end
if ~isequal(size(x_ref), [d_x, T, M])
    error('x_ref must be d_x×T×M after expansion. Got %s.', mat2str(size(x_ref)));
end

% --- force reference path slot j=1 at t=1 ---
X(:,1,:) = x_ref(:,1,:);

% --- preallocate internal storage needed for trajectory construction ---
% We must keep particles & logw for all t to sample a trajectory.
particles_int = zeros(d_x, N, T, M, 'like', X);
ancestors_int = zeros(N, T, M, 'uint32');
logw_int      = -inf(N, T, M);

% ===================== main forward pass =====================
for t = 1:T
    % store pre-resampling cloud
    particles_int(:, :, t, :) = X;

    % log-likelihoods: 1×N×M
    ll = g(y(:, t), X, g_pars, t);
    if ~isequal(size(ll), [1 N M])
        error('g must return 1×N×M at t=%d; got %s.', t, mat2str(size(ll)));
    end

    % ---- normalize log-weights per filter (stable log-sum-exp) ----
    L = squeeze(permute(ll, [2 3 1]));      % N × M
    Lmax = max(L, [], 1);
    shifted = L - Lmax;
    W = exp(shifted);                        % N × M
    S = sum(W, 1);                           % 1 × M
    allInf = (S == 0);

    w_norm = zeros(N, M);
    if any(~allInf), w_norm(:, ~allInf) = W(:, ~allInf) ./ S(1, ~allInf); end
    if any(allInf),  w_norm(:,  allInf) = 1 / N;                           end

    % store normalized log-weights
    logw = -inf(N, M);
    if any(~allInf)
        denom = Lmax(1, ~allInf) + log(S(1, ~allInf));
        logw(:, ~allInf) = bsxfun(@minus, L(:, ~allInf), denom);
    end
    if any(allInf), logw(:, allInf) = -log(N); end
    logw_int(:, t, :) = reshape(logw, N, 1, M);

    % ---- multinomial resampling per filter; fix reference ancestor ----
    A = multinomial_resample_sorted(w_norm, streams);   % N×M
    A(1, :) = 1;                                        % keep ref in slot 1
    ancestors_int(:, t, :) = uint32(A);

    % ---- propagate to t+1 and re-impose reference in slot 1 ----
    if t < T
        X_post = select_by_ancestors(X, A);             % d_x×N×M
        X = trans_dist_samp(X_post, trans_pars, t+1);
        if ~isequal(size(X), [d_x, N, M])
            error('trans_dist_samp returned %s, expected d_x×N×M.', mat2str(size(X)));
        end
        X(:,1,:) = x_ref(:,t+1,:);                      % enforce conditional path at t+1
    end
end
% =============================================================

% ---- sample trajectory per filter (mode: "ancestors" or "backward") ----
sampled_idx  = zeros(1, M);
sampled_path = zeros(d_x, T, M, 'like', X);

% final weights
logw_T = reshape(logw_int(:, T, :), N, M);   % N×M (already log)
for m = 1:M
    % draw terminal index
    j = draw_from_logw_np(logw_T(:, m).', streams{m});
    sampled_idx(m) = j;

    if traj_mode == "ancestors"
        % backtrack via ancestors (note: use t-1)
        for t = T:-1:1
            sampled_path(:, t, m) = particles_int(:, j, t, m);
            if t > 1
                j = double(ancestors_int(j, t-1, m));
            end
        end

    elseif traj_mode == "backward"
        % backward simulation using trans_logpdf
        sampled_path(:, T, m) = particles_int(:, j, T, m);
        for t = T-1:-1:1
            x_next = sampled_path(:, t+1, m);              % d_x×1
            X_t    = particles_int(:, :, t, m);            % d_x×N
            lw_t   = (logw_int(:, t, m)).';                % 1×N
            lf     = trans_logpdf(x_next, X_t, trans_pars, t+1);   % 1×N
            if ~isequal(size(lf), [1, N])
                error('trans_logpdf must return 1×N at t=%d.', t+1);
            end
            logK = lw_t + lf;                               % 1×N
            j = draw_from_logw_np(logK, streams{m});
            sampled_path(:, t, m) = X_t(:, j);
        end
    else
        error('traj_mode must be "ancestors" or "backward".');
    end
end

% ----------------- package outputs (respect store_* flags) -----------------
out.sampled_path = sampled_path;
out.sampled_idx  = sampled_idx;
out.sampled_mode = char(traj_mode);

if store_particles, out.particles = particles_int; end
if store_ancestors, out.ancestors = ancestors_int; end
if store_logw,     out.logw      = logw_int;      end

end % cpf_parallel






function out = cpf_parallel( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    x_ref, seed, ...
    traj_mode, trans_logpdf, ...
    store_particles, store_ancestors, store_logw)
%CPF_PARALLEL  Conditional bootstrap PF with selectable trajectory sampler (parallelized per filter).
%
% [same header & notes as your version]

% ----------------- defaults & input handling -----------------
if nargin < 12 || isempty(seed),       seed = 12345;        end
if nargin < 13 || isempty(traj_mode),  traj_mode = "backward"; end
traj_mode = lower(string(traj_mode));

if nargin < 14, trans_logpdf = []; end
if traj_mode == "backward" && ~isa(trans_logpdf,'function_handle')
    error('traj_mode="backward" requires a valid trans_logpdf function handle.');
end

if nargin < 15 || isempty(store_particles),  store_particles  = true; end
if nargin < 16 || isempty(store_ancestors),  store_ancestors  = true; end
if nargin < 17 || isempty(store_logw),       store_logw       = true; end

% ----------------- basic checks -----------------
[~, T_chk] = size(y);
if T_chk ~= T, error('size(y,2)=%d does not match T=%d.', T_chk, T); end

% --- initialize particles at t=1 (pre-resampling) ---
X = in_dist_samp(in_pars, N, M);           % d_x × N × M
[d_x, N_chk, M_chk] = size(X);
if N_chk ~= N || M_chk ~= M
    error('in_dist_samp returned [%d %d %d], expected d_x×N×M with N=%d, M=%d.', ...
          d_x, N_chk, M_chk, N, M);
end

% --- accept x_ref as d_x×T or d_x×T×1 and expand to d_x×T×M ---
if ndims(x_ref) == 2
    if ~isequal(size(x_ref), [d_x, T])
        error('x_ref must be d_x×T (or d_x×T×M). Got %s.', mat2str(size(x_ref)));
    end
    x_ref = reshape(x_ref, d_x, T, 1);
end
if size(x_ref,3) == 1 && M > 1
    x_ref = repmat(x_ref, 1, 1, M);
end
if ~isequal(size(x_ref), [d_x, T, M])
    error('x_ref must be d_x×T×M after expansion. Got %s.', mat2str(size(x_ref)));
end

% --- force reference path slot j=1 at t=1 ---
X(:,1,:) = x_ref(:,1,:);

% --- preallocate internal storage needed for trajectory construction ---
particles_int = zeros(d_x, N, T, M, 'like', X);
ancestors_int = zeros(N, T, M, 'uint32');
logw_int      = -inf(N, T, M);

% ===================== main forward pass =====================
for t = 1:T
    % store pre-resampling cloud
    particles_int(:, :, t, :) = X;

    % allocate per-step containers (sliced by m for parfor)
    A_step    = zeros(N, M, 'uint32');
    logw_step = -inf(N, M);
    X_next    = zeros(d_x, N, M, 'like', X);   % only used if t<T

    % ===== BEGIN: parallel per-filter work =====
    parfor m = 1:M
        % deterministic RNG seed per (t,m)
        % (add a large stride per t so each time has an independent stream)
        seed_tm = seed + (m-1) + 100000*(t-1);
        try
            rng(seed_tm, 'Threefry'); %#ok<RNGB>
        catch
            rng(seed_tm, 'twister');  %#ok<RNGB> % fallback if Threefry not available
        end

        % ---- log-likelihoods for this filter (shape it to N×1) ----
        ll_m = g(y(:, t), X(:, :, m), g_pars, t);   % expected 1×N or 1×N×1
        L = ll_m(:);                                % N×1

        % ---- normalize (log-sum-exp) ----
        Lmax = max(L);
        W = exp(L - Lmax);
        S = sum(W);
        if S == 0 || ~isfinite(S)
            w = ones(N,1) / N;
            logw_m = (-log(N)) * ones(N,1);
        else
            w = W / S;
            logw_m = L - (Lmax + log(S));
        end
        logw_step(:, m) = logw_m;

        % ---- multinomial resampling (sorted uniforms) ----
        u = sort(rand(N,1));
        c = cumsum(w);
        Ai = zeros(N,1,'uint32');
        i = 1; j = 1;
        while i <= N
            while u(i) > c(j), j = j + 1; end
            Ai(i) = uint32(j); i = i + 1;
        end
        Ai(1) = uint32(1);   % keep reference in slot 1
        A_step(:, m) = Ai;

        % ---- propagate to t+1 (form post-resample cloud locally) ----
        if t < T
            X_post_m = X(:, double(Ai), m);                  % d_x×N
            Xn = trans_dist_samp(X_post_m, trans_pars, t+1); % d_x×N
            Xn(:,1) = x_ref(:, t+1, m);                      % enforce conditional path
            X_next(:, :, m) = Xn;
        end
    end
    % ===== END: parallel per-filter work =====

    % collect this step's results
    logw_int(:, t, :)      = reshape(logw_step, N, 1, M);
    ancestors_int(:, t, :) = A_step;

    if t < T
        X = X_next;  % move to next time
    end
end
% =============================================================

% ---- sample trajectory per filter (mode: "ancestors" or "backward") ----
sampled_idx  = zeros(1, M);
sampled_path = zeros(d_x, T, M, 'like', X);

% final weights
wT = exp(logw_int(:, T, :));    % N × 1 × M
wT = reshape(wT, N, M);         % N × M

% ===== BEGIN: parallel backtrace per filter =====
parfor m = 1:M
    % local RNG for terminal draw (keep deterministic per m)
    seed_Tm = seed + (m-1) + 100000*T;
    try
        rng(seed_Tm, 'Threefry'); %#ok<RNGB>
    catch
        rng(seed_Tm, 'twister');  %#ok<RNGB>
    end

    % draw terminal index from final weights
    jT = draw_from_logw(log(wT(:, m).'+realmin));  % takes 1×N logw row
    sampled_idx(m) = jT;

    % build the whole slice locally, then assign once
    sp = zeros(d_x, T, 'like', X);   % local buffer for sampled_path(:,:,m)

    if traj_mode == "ancestors"
        j = jT;
        for t = T:-1:1
            sp(:, t) = particles_int(:, j, t, m);
            %sampled_path(:, t, m) = particles_int(:, j, t, m);
            if t > 1
                j = double(ancestors_int(j, t-1, m));
            end
        end

    elseif traj_mode == "backward"
        j = jT;
        sp(:, T) = particles_int(:, j, T, m);
        %sampled_path(:, T, m) = particles_int(:, j, T, m);
        for t = T-1:-1:1
            x_next = sp(:, t+1);         % d_x×1
            %x_next = sampled_path(:, t+1, m);         % d_x×1
            X_t    = particles_int(:, :, t, m);       % d_x×N
            lw_t   = (logw_int(:, t, m)).';           % 1×N
            lf     = trans_logpdf(x_next, X_t, trans_pars, t+1); % 1×N
            if ~isequal(size(lf), [1, N])
                error('trans_logpdf must return 1×N at t=%d.', t+1);
            end
            logK = lw_t + lf;                          % 1×N
            j = draw_from_logw(logK);
            sp(:, t) = X_t(:, j);
            %sampled_path(:, t, m) = X_t(:, j);
        end
    else
        error('traj_mode must be "ancestors" or "backward".');
    end
    sampled_path(:, :, m) = sp;
end
% ===== END: parallel backtrace per filter =====

% ----------------- package outputs (respect store_* flags) -----------------
out.sampled_path = sampled_path;
out.sampled_idx  = sampled_idx;
out.sampled_mode = char(traj_mode);

if store_particles, out.particles = particles_int; end
if store_ancestors, out.ancestors = ancestors_int; end
if store_logw,     out.logw      = logw_int;      end

end % cpf_parallel


% ----------------------- helpers -----------------------
function j = draw_from_logw(logw_row)
% Draw index (1..N) from a 1×N vector of (un-normalized) log-weights.
    if size(logw_row,1) ~= 1, logw_row = logw_row(:).'; end
    Lmax = max(logw_row);
    x = exp(logw_row - Lmax);
    s = sum(x);
    if s == 0 || ~isfinite(s)
        % fallback: uniform
        N = numel(logw_row);
        j = 1 + floor(rand(1) * N);
        return;
    end
    p = x / s;
    u = rand(1);
    c = cumsum(p);
    j = find(u <= c, 1, 'first');
end


function j = draw_from_logw_np(logw_row, stream)
% Draw index (1..N) from a 1×N vector of log-weights (unnormalized is OK).
% Stable: subtract max, exp, normalize, CDF, one uniform.
    if size(logw_row,1) ~= 1
        logw_row = logw_row(:).';
    end
    Lmax = max(logw_row);
    x = exp(logw_row - Lmax);
    s = sum(x);
    if s == 0 || ~isfinite(s)
        % fallback: uniform
        j = 1 + floor(rand(stream,1) * numel(logw_row));
        return;
    end
    p = x / s;
    u = rand(stream,1);
    c = cumsum(p);
    j = find(u <= c, 1, 'first');
end

% ----------------------- helpers -----------------------

%% ======================= Local function: RTS smoother =======================
function [m_f, P_f, m_s, P_s] = rts_smoother_1d(y, rho, q, r, H, m0, P0)
% Forward: Kalman filter (update at t, then predict to t+1)
T = size(y,2);
m_f = zeros(1,T); P_f = zeros(1,T);
m = m0; P = P0;
for t = 1:T
    % Update with y_t
    S = H*P*H' + r;                 % innovation variance
    K = (P*H') / S;                 % gain
    innov = y(1,t) - H*m;
    m = m + K*innov;
    P = (1 - K*H)*P;
    m_f(t) = m; P_f(t) = P;
    % Predict to t+1
    if t < T
        m = rho*m;
        P = rho*P*rho + q;
    end
end

% Backward: RTS smoothing
m_s = zeros(1,T); P_s = zeros(1,T);
m_s(T) = m_f(T); P_s(T) = P_f(T);
for t = T-1:-1:1
    % Predict stats from t to t+1 (using filtered at t)
    m_pred = rho * m_f(t);
    P_pred = rho * P_f(t) * rho + q;
    % Smoother gain
    C = (P_f(t) * rho) / P_pred;
    % Smoothed mean/var
    m_s(t) = m_f(t) + C * (m_s(t+1) - m_pred);
    P_s(t) = P_f(t) + C^2 * (P_s(t+1) - P_pred);
end
end

function out = pgibbs_run_init_pfmean( ...
    y, T, N, M, B, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    seed0, ...
    cpf_choice, ...
    traj_mode, trans_logpdf, ...
    store_particles, store_ancestors, store_logw, ...
    N_init, init_mode)

% Defaults
if nargin < 13 || isempty(cpf_choice),   cpf_choice   = "cpf_parallel"; end
if nargin < 14 || isempty(traj_mode),    traj_mode    = "ancestors";    end
if nargin < 15, trans_logpdf = []; end
if nargin < 16 || isempty(store_particles), store_particles = false; end
if nargin < 17 || isempty(store_ancestors), store_ancestors = false; end
if nargin < 18 || isempty(store_logw),      store_logw      = false; end
if nargin < 19 || isempty(N_init),          N_init          = N;     end
if nargin < 20 || isempty(init_mode),       init_mode       = "weighted"; end

cpf_choice = lower(string(cpf_choice));
traj_mode  = lower(string(traj_mode));
init_mode  = lower(string(init_mode));

if ~(cpf_choice=="cpf" || cpf_choice=="cpf_parallel")
    error('cpf_choice must be "cpf" or "cpf_parallel".');
end
if traj_mode=="backward" && ~isa(trans_logpdf,'function_handle')
    error('traj_mode="backward" requires trans_logpdf.');
end
if size(y,2) ~= T, error('size(y,2)=%d ≠ T=%d', size(y,2), T); end

% Probe d_x
X0 = in_dist_samp(in_pars, max(1,N), max(1,M));
d_x = size(X0,1);

% ---- 1) Build M independent initial paths from PF(s) ----
x_init_paths = pf_weighted_mean_paths( ...
    y, T, N_init, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, seed0, init_mode);       % d_x×T×M

% ---- 2) PG over B links, store B+1 paths ----
X_paths = zeros(d_x, T, M, B+1, 'like', x_init_paths);
X_paths(:,:,:,1) = x_init_paths;

cpf_fun = str2func(char(cpf_choice));
time_per_iter = zeros(1, B);

x_ref = x_init_paths;
for b = 1:B
    t0 = tic;
    out_cpf = cpf_fun(y, T, N, M, ...
                      in_dist_samp, in_pars, ...
                      trans_dist_samp, trans_pars, ...
                      g, g_pars, ...
                      x_ref, seed0 + (b-1), ...
                      traj_mode, trans_logpdf, ...
                      store_particles, store_ancestors, store_logw);
    time_per_iter(b) = toc(t0);

    X_paths(:,:,:,b+1) = out_cpf.sampled_path;
    x_ref = out_cpf.sampled_path;   % condition next step on current sample
end

% ---- 3) package ----
out.X_paths       = X_paths;           % d_x×T×M×(B+1)
out.time_per_iter = time_per_iter;
out.init_from     = sprintf('PF %s mean (N_init=%d, M=%d)', char(init_mode), N_init, M);
out.settings = struct('T',T,'N',N,'M',M,'B',B, ...
                      'cpf_choice',char(cpf_choice), ...
                      'traj_mode',char(traj_mode), ...
                      'seed0',seed0, ...
                      'N_init',N_init, 'init_mode',char(init_mode), ...
                      'store_particles',store_particles, ...
                      'store_ancestors',store_ancestors, ...
                      'store_logw',store_logw);
end


% ===== helper: run a PF and return the weighted filtered mean path (d_x×T) =====
function x_init_paths = pf_weighted_mean_paths( ...
    y, T, N_init, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, seed, init_mode)

% init_mode: "weighted" (default) or "resampled"

if nargin < 13 || isempty(init_mode), init_mode = "weighted"; end
init_mode = lower(string(init_mode));

% (Optional) make g robust to M=1 by forcing 1×N×M shape
g_fix = @(yt,Xt,p,t) reshape( ...
           g(yt,Xt,p,t), 1, size(Xt,2), size(Xt,3));

% Run one PF with M filters; we need particles & logw
out_pf = pf_parallel(y, T, N_init, M, ...
                     in_dist_samp, in_pars, ...
                     trans_dist_samp, trans_pars, ...
                     g_fix, g_pars, seed);

if ~isfield(out_pf,'particles') || ~isfield(out_pf,'logw')
    error('pf_parallel must return .particles (d_x×N×T×M) and .logw (N×T×M).');
end

X  = out_pf.particles;   % d_x × N × T × M (pre-resampling clouds)
lw = out_pf.logw;        % N × T × M (normalized log-weights, log)
w  = exp(lw);            % N × T × M

[d_x, N, T, M] = size(X);

switch init_mode
case "weighted"
    % x_init(:,t,m) = sum_i w(i,t,m) * X(:,i,t,m)

    tmp = sum( X .* reshape(w, 1, N, T, M), 2);  % d_x × 1 × T × M
    x_init_paths = reshape(tmp, size(X,1), T, M);  % d_x × T × M

case "resampled"
    % Build a resampled cloud per (t,m), then average (adds variance but uses ancestors idea)
    x_init_paths = zeros(d_x, T, M, 'like', X);
    for m = 1:M
        for t = 1:T
            % multinomial draw N indices from weights w(:,t,m)
            pm = w(:,t,m) ./ max(sum(w(:,t,m)), realmin);
            u  = sort(rand(N,1));
            c  = cumsum(pm);
            idx = zeros(N,1,'uint32');
            i=1; j=1;
            while i<=N
                while u(i)>c(j), j=j+1; end
                idx(i)=uint32(j); i=i+1;
            end
            Xrt = X(:, double(idx), t, m);      % resampled cloud
            x_init_paths(:, t, m) = mean(Xrt, 2);
        end
    end

otherwise
    error('init_mode must be "weighted" or "resampled".');
end
end

function [score, parts] = score_gaussian_ssm(y, theta, q, r, S0)
% SCORE_GAUSSIAN_SSM  Gradient (score) of the log-likelihood for a 1D LGSSM.
% Model:
%   X1 ~ N(0, S0)
%   Xt | X_{t-1} ~ N(theta * X_{t-1}, q)
%   Yt | Xt ~ N(Xt, r)
%
% Inputs
%   y      : 1×T observations
%   theta  : scalar (state coefficient)
%   q      : scalar > 0 (state var)
%   r      : scalar > 0 (obs var)
%   S0     : scalar > 0 (initial var)
%
% Outputs
%   score  : struct with fields dtheta, dq, dr, dS0
%   parts  : struct with useful internals (filtered, smoothed, cross-cov, etc.)

    y  = y(:)';                 % 1×T
    T  = size(y,2);
    eps_small = 1e-12;

    % ---------- Forward: Kalman filter (update at t, then predict to t+1)
    m_f = zeros(1,T);  P_f = zeros(1,T);
    m_pred = zeros(1,T);  P_pred = zeros(1,T);   % store P_{t|t-1} (shifted)
    % prior for t=1 (before seeing y1)
    m = 0;
    P = S0;
    for t = 1:T
        % innovation update at t
        S_t = P + r;                          % 1×1
        K   = P / S_t;
        innov = y(t) - m;
        m = m + K * innov;
        P = (1 - K) * P;
        m_f(t) = m;  P_f(t) = P;

        % prediction to t+1
        if t < T
            m_pred(t+1) = theta * m;
            P_pred(t+1) = theta^2 * P + q;
            m = m_pred(t+1);
            P = P_pred(t+1);
        end
    end

    % ---------- Backward: RTS smoother + lag-one covariance
    m_s = zeros(1,T);  P_s = zeros(1,T);
    J   = zeros(1,T-1);           % smoother gains J_t for t=1..T-1
    C_lag = zeros(1,T);           % Cov(X_t, X_{t-1} | y), defined for t>=2

    m_s(T) = m_f(T);  P_s(T) = P_f(T);
    for t = T-1:-1:1
        % P_pred(t+1) is prediction variance from t to t+1
        Pp = max(P_pred(t+1), eps_small);
        J(t) = (P_f(t) * theta) / Pp;

        % smooth
        m_s(t) = m_f(t) + J(t) * (m_s(t+1) - theta*m_f(t));
        P_s(t) = P_f(t) + J(t)^2 * (P_s(t+1) - Pp);

        % lag-one covariance: Cov(X_t, X_{t-1} | y), here for index (t) vs (t-1)
        % Use Σ_{t-1,t}^s = J(t-1) * Σ_t^s  ⇒ Cov(X_t, X_{t-1}) = J(t-1) * P_s(t)
        % We'll fill it after loop for t>=2 using J(t-1)
    end
    for t = 2:T
        C_lag(t) = J(t-1) * P_s(t);   % Cov(X_t, X_{t-1} | y)
    end

    % ---------- Expectations needed for the Fisher score
    % E[X_t] = m_s(t);  Var[X_t] = P_s(t);
    % E[X_t^2] = P_s(t) + m_s(t)^2
    EX2  = P_s + m_s.^2;
    % E[X_t X_{t-1}] = Cov + mean product
    EXXt = C_lag + m_s .* [0, m_s(1:end-1)];   % first entry unused (t=1)

    % ---------- Score components
    % (1) wrt theta:  sum_{t=2..T} (1/q) E[(X_t - theta X_{t-1}) X_{t-1}]
    %     = (1/q) sum_{t=2..T} (E[X_t X_{t-1}] - theta E[X_{t-1}^2])
    E_XtXm1   = EXXt(2:end);
    E_Xm1sq   = EX2(1:end-1);
    dtheta = (1/max(q,eps_small)) * sum( E_XtXm1 - theta * E_Xm1sq );

    % (2) wrt q:  -(T-1)/(2q) + (1/(2q^2)) sum_{t=2..T} E[(X_t - theta X_{t-1})^2]
    % E[(X_t - theta X_{t-1})^2] = E[X_t^2] - 2theta E[X_t X_{t-1}] + theta^2 E[X_{t-1}^2]
    E_res2 = EX2(2:end) - 2*theta*E_XtXm1 + theta^2 * E_Xm1sq;
    dq = -(T-1)/(2*max(q,eps_small)) + 0.5 * sum(E_res2) / max(q,eps_small)^2;

    % (3) wrt r:  -T/(2r) + (1/(2r^2)) sum_t E[(Y_t - X_t)^2]
    % E[(Y_t - X_t)^2] = (y - m_s).^2 + P_s
    E_meas2 = (y - m_s).^2 + P_s;
    dr = -T/(2*max(r,eps_small)) + 0.5 * sum(E_meas2) / max(r,eps_small)^2;

    % (4) wrt S0 (initial variance):  -1/(2S0) + (1/(2S0^2)) E[X_1^2]
    dS0 = -1/(2*max(S0,eps_small)) + 0.5 * EX2(1) / max(S0,eps_small)^2;

    % ---------- Package
    score = struct('dtheta', dtheta, 'dq', dq, 'dr', dr, 'dS0', dS0);

    if nargout > 1
        parts = struct();
        parts.m_f = m_f; parts.P_f = P_f;
        parts.m_s = m_s; parts.P_s = P_s;
        parts.J = J; parts.P_pred = P_pred;
        parts.C_lag = C_lag;            % Cov(X_t, X_{t-1} | y), t>=2
        parts.EX2 = EX2; parts.EXXt = EXXt;
        parts.E_res2 = E_res2; parts.E_meas2 = E_meas2;
    end
end


% function for the gradients of the logs of the ssm.
function g = grad_log_g_gauss(y_t, x_t, obs_pars, ~)
% y_t, x_t are scalars here (1×1), obs_pars.R = r
r = obs_pars.R;
res2 = (y_t - x_t).^2;
g = -0.5/r + 0.5*res2/(r^2);   % scalar gradient wrt r
end

function g = grad_log_f_gauss(x_t, x_tm1, trans_pars, ~)
% trans_pars: .theta, .q
theta = trans_pars.theta; q = trans_pars.q;
res   = x_t - theta*x_tm1;
dtheta = (res * x_tm1) / q;
dq     = -0.5/q + 0.5*(res.^2)/(q^2);
g = [dtheta; dq];               % 2×1
end

function g = grad_log_p1_gauss(x1, init_pars)
% init_pars.S0
S0 = init_pars.S0;
g  = -0.5/S0 + 0.5*(x1.^2)/(S0^2);  % scalar
end

function out = score_from_paths( ...
    y, X_paths, ...
    grad_log_g, grad_log_f, grad_log_p1, ...
    obs_pars, trans_pars, init_pars, ...
    paths_sel, weights, mode)

% SCORE_FROM_PATHS  Score (gradient) from PG-sampled paths.
% Supports:
%   mode="sequential"         (generic; uses callbacks as-is)
%   mode="vectorized_gauss1d" (fast path for 1D Gaussian AR(1): r, theta, q, S0)
%   mode="vectorized_callbacks"
% Inputs
%   y            : d_y × T
%   X_paths      : d_x × T × M × B1
%   grad_log_*   : callbacks (only used in "sequential" mode)
%   *_pars       : structs with parameters (for gauss1d: obs_pars.R=r; trans_pars.theta,q; init_pars.S0)
%   paths_sel    : logical M×B1 (optional; default all true)
%   weights      : P×1 (optional; default uniform)
%   mode         : "sequential" (default) | "vectorized_gauss1d"
%
% Output
%   out.avg_obs   : gradient wrt observation parameters
%   out.avg_trans : gradient wrt transition  parameters
%   out.avg_init  : gradient wrt initial     parameters
%   out.avg_total : concatenation [obs; trans; init]

    if nargin < 9 || isempty(paths_sel)
        paths_sel = true(size(X_paths,3), size(X_paths,4));
    end
    if nargin < 10 || isempty(weights)
        % filled after we know P
    end
    if nargin < 11 || isempty(mode)
        mode = "sequential";
    else
        mode = string(mode);
    end

    [d_x, T, M, B1] = size(X_paths); %#ok<NASGU>
    idx = find(paths_sel(:));
    P   = numel(idx);
    if P == 0
        error('paths_sel selects zero paths.');
    end

    % Extract selected paths to d_x × T × P
    Xp = zeros(size(X_paths,1), size(X_paths,2), P, 'like', X_paths);
    [mm, bb] = ind2sub([size(X_paths,3), size(X_paths,4)], idx);
    for k = 1:P
        Xp(:,:,k) = X_paths(:,:,mm(k), bb(k));
    end

    % weights
    if nargin < 10 || isempty(weights)
        weights = ones(P,1, 'like', Xp) / P;
    else
        weights = weights(:) / sum(weights);
    end

    switch lower(mode)
        case "sequential"
            % ----------------- generic, model-agnostic -----------------
            g_obs   = 0;    % size set by first callback return
            g_trans = 0;
            g_init  = 0;

            for k = 1:P
                x = Xp(:,:,k);             % d_x × T
                g0 = grad_log_p1(x(:,1), init_pars);
                gt = 0; go = 0;
                for t = 1:T
                    go = go + grad_log_g(y(:,t), x(:,t), obs_pars, t);
                    if t >= 2
                        gt = gt + grad_log_f(x(:,t), x(:,t-1), trans_pars, t);
                    end
                end
                w = weights(k);
                g_init  = g_init  + w * g0;
                g_trans = g_trans + w * gt;
                g_obs   = g_obs   + w * go;
            end

        case "vectorized_callbacks"
    % ================= vectorized via user callbacks =================
    % Expect the callbacks to accept all paths at once and return
    % per-time, per-path contributions to be reduced here.
    %
    % Required vectorized signatures:
    %   Gg = grad_log_g( y, Xp, obs_pars )
    %       y   : d_y × T                  (shared)
    %       Xp  : d_x × T × P              (all selected paths)
    %       Gg  : p_obs × T × P            (per-time, per-path contributions)
    %
    %   Gf = grad_log_f( Xp, trans_pars )
    %       Xp  : d_x × T × P
    %       Gf  : p_tr  × (T-1) × P        (per-time (t=2..T), per-path)
    %
    %   G0 = grad_log_p1( X1, init_pars )
    %       X1  : d_x × 1 × P              (just the initial state per path)
    %       G0  : p_init × P                (per-path)
    %
    % Notes:
    % - We sum across time inside this branch, then average across paths
    %   using 'weights' (P×1).
    % - Shapes p_obs, p_tr, p_init can be any positive integers.

    % 1) Observation term: p_obs×T×P  -> sum over T -> p_obs×P
    Gg = grad_log_g(y, Xp, obs_pars);                 % p_obs×T×P
    if ndims(Gg) ~= 3 || size(Gg,2) ~= T || size(Gg,3) ~= P
        error('grad_log_g must return p_obs×T×P; got %s.', mat2str(size(Gg)));
    end
    Gg_sum = squeeze(sum(Gg, 2));                     % p_obs×P

    % 2) Transition term: p_tr×(T-1)×P -> sum over (T-1) -> p_tr×P
    Gf = grad_log_f(Xp, trans_pars);                  % p_tr×(T-1)×P
    if ndims(Gf) ~= 3 || size(Gf,2) ~= (T-1) || size(Gf,3) ~= P
        error('grad_log_f must return p_tr×(T-1)×P; got %s.', mat2str(size(Gf)));
    end
    Gf_sum = squeeze(sum(Gf, 2));                     % p_tr×P

    % 3) Initial term: p_init×P (already per-path)
    X1 = Xp(:,1,:);                                   % d_x×1×P
    G0 = grad_log_p1(X1, init_pars);                  % p_init×P
    if ~ismatrix(G0) || size(G0,2) ~= P
        error('grad_log_p1 must return p_init×P; got %s.', mat2str(size(G0)));
    end

    % 4) Weighted averages across paths (broadcast weights: 1×P)
    wrow = reshape(weights, 1, P);                    % 1×P

    % observation gradient: p_obs×P  • w -> p_obs×1
    g_obs   = Gg_sum * weights;                       % p_obs×1

    % transition gradient: p_tr×P    • w -> p_tr×1
    g_trans = Gf_sum * weights;                       % p_tr×1

    % initial gradient:    p_init×P  • w -> p_init×1
    g_init  = G0     * weights;                       % p_init×1


        case "vectorized_gauss1d"
            % ----------------- fast path: 1D Gaussian AR(1) -----------------
            % Checks
            if size(Xp,1) ~= 1
                error('vectorized_gauss1d requires d_x = 1.');
            end
            if ~isfield(obs_pars,'R') || ~isfield(trans_pars,'theta') || ~isfield(trans_pars,'q') || ~isfield(init_pars,'S0')
                error('vectorized_gauss1d expects fields: obs_pars.R, trans_pars.theta,q, init_pars.S0.');
            end

            r     = obs_pars.R;
            theta = trans_pars.theta; q = trans_pars.q;
            S0    = init_pars.S0;

            % shapes: squeeze to T×P (since d_x=1)
            Xall = squeeze(Xp);           % T × P
            if isrow(Xall), Xall = Xall.'; end
            yrow = y(:).';                % 1 × T

            % --- observation term (dr)
            diff_yx = yrow.' - Xall;      % T × P
            dr_each = -0.5./r + 0.5*(diff_yx.^2)./(r.^2);   % T × P
            dr_k    = sum(dr_each, 1);                        % 1 × P
            g_obs   = sum(weights.' .* dr_k);                % scalar

            % --- transition term (dtheta, dq)
            x_t   = Xall(2:end, :);        % (T-1) × P
            x_tm1 = Xall(1:end-1, :);      % (T-1) × P
            res   = x_t - theta .* x_tm1;  % (T-1) × P

            dtheta_each = (res .* x_tm1) ./ q;               % (T-1) × P
            dq_each     = -0.5./q + 0.5*(res.^2)./(q.^2);    % (T-1) × P

            dtheta_k = sum(dtheta_each, 1);                  % 1 × P
            dq_k     = sum(dq_each, 1);                      % 1 × P

            dtheta = sum(weights.' .* dtheta_k);             % scalar
            dq     = sum(weights.' .* dq_k);                 % scalar
            g_trans = [dtheta; dq];

            % --- initial term (dS0)
            x1     = Xall(1, :);                             % 1 × P
            dS0_k  = -0.5./S0 + 0.5*(x1.^2)./(S0.^2);        % 1 × P
            g_init = sum(weights.' .* dS0_k);                % scalar

        otherwise
            error('Unknown mode "%s". Use "sequential" or "vectorized_gauss1d".', mode);
    end

    out = struct();
    out.avg_obs   = g_obs;
    out.avg_trans = g_trans;
    out.avg_init  = g_init;
    % concatenate in the natural order used elsewhere: [dr; dtheta; dq; dS0] for gauss1d
    try
        out.avg_total = [g_obs; g_trans; g_init];
    catch
        % If shapes don't concatenate (e.g., model-specific vectors), skip total
        out.avg_total = [];
    end
end
%%
function out = score_gaussian_from_paths(y, X_paths, theta, q, r, S0, exclude_first_col)
% y: 1×T, X_paths: 1×T×M×B1
if nargin < 7, exclude_first_col = false; end
obs_pars   = struct('R', r);
trans_pars = struct('theta', theta, 'q', q);
init_pars  = struct('S0', S0);

[M, B1] = deal(size(X_paths,3), size(X_paths,4));
paths_sel = true(M,B1);
if exclude_first_col, paths_sel(:,1) = false; end

out = score_from_paths(y, X_paths, ...
        @grad_log_g_gauss, @grad_log_f_gauss, @grad_log_p1_gauss, ...
        obs_pars, trans_pars, init_pars, paths_sel, []);
end


%% ===== helper: one run =====
function [enorm, ecomp, tsec] = one_run_score_err( ...
    y,T,N,M,B, ...
    in_dist,in_pars,trans,tr_pars,g,g_pars, ...
    seed0, ...
    cpf_choice,traj_mode,trans_logpdf, ...
    store_parts,store_anc,store_logw, ...
    N_init,init_mode, g_anal)

t0 = tic;
out_pg = pgibbs_run_init_pfmean( ...
    y, T, N, M, B, ...
    in_dist, in_pars, ...
    trans, tr_pars, ...
    g, g_pars, ...
    seed0, ...
    cpf_choice, ...
    traj_mode, trans_logpdf, ...
    store_parts, store_anc, store_logw, ...
    N_init, init_mode);
tsec = toc(t0);

X_paths = out_pg.X_paths;                  % 1×T×M×(B+1)
S_mc    = score_gaussian_from_paths_vectorized(y, X_paths, tr_pars.theta, tr_pars.q, g_pars.R, in_pars.Sigma, true);
g_mc    = [S_mc.avg_obs; S_mc.avg_trans; S_mc.avg_init];  % [dr; dtheta; dq; dS0]
diff    = g_mc - g_anal;
enorm   = norm(diff,2);
ecomp   = abs(diff).';
end

%%
% Observation gradient (wrt r): returns p_obs×T×P with p_obs=1
function Gg = grad_log_g_gauss_vec(y, Xp, obs_pars)
% y  : 1×T
% Xp : 1×T×P
% Gg : 1×T×P  (per-time, per-path)
r = obs_pars.R;
Y = reshape(y, 1, numel(y), 1);              % 1×T×1
diff_yx = Y - Xp;                            % 1×T×P
Gg = -0.5./r + 0.5*(diff_yx.^2)./(r.^2);     % 1×T×P
end

% Transition gradient (wrt [theta; q]): returns p_tr×(T-1)×P with p_tr=2
function Gf = grad_log_f_gauss_vec(Xp, trans_pars)
% Xp : 1×T×P
% Gf : 2×(T-1)×P  (row 1: dtheta, row 2: dq)
theta = trans_pars.theta;
q     = trans_pars.q;

x_tm1 = Xp(:, 1:end-1, :);                   % 1×(T-1)×P
x_t   = Xp(:, 2:end,   :);                   % 1×(T-1)×P
res   = x_t - theta .* x_tm1;                % 1×(T-1)×P

dtheta = (res .* x_tm1) ./ q;                % 1×(T-1)×P
dq     = -0.5./q + 0.5*(res.^2)./(q.^2);     % 1×(T-1)×P

Gf = cat(1, dtheta, dq);                      % 2×(T-1)×P
end

% Initial gradient (wrt S0): returns p_init×P with p_init=1
function G0 = grad_log_p1_gauss_vec(X1, init_pars)
% X1 : 1×1×P   (initial state per path)
% G0 : 1×P
S0 = init_pars.S0;
x1 = squeeze(X1);                             % 1×P
G0 = -0.5./S0 + 0.5*(x1.^2)./(S0.^2);        % 1×P
end


%% ===== scoring helpers (same as earlier) =====

function out = score_gaussian_from_paths_vectorized(y, X_paths, theta, q, r, S0, paths_sel, weights, exclude_first_col)
% Vectorized score over ALL selected paths (no per-path loops).
% y        : 1×T
% X_paths  : 1×T×M×B1
% theta,q,r,S0 : scalars
% paths_sel: logical M×B1 (optional; default all true)
% weights  : P×1 (optional; default uniform over selected)
% exclude_first_col : if true, ignore initializer column (b=1)
%
% Returns:
%   out.avg_obs    (scalar: dr)
%   out.avg_trans  (2×1: [dtheta; dq])
%   out.avg_init   (scalar: dS0)
%   out.avg_total  (4×1: [dr; dtheta; dq; dS0])

    if nargin < 7 || isempty(paths_sel)
        paths_sel = true(size(X_paths,3), size(X_paths,4));
    end
    if nargin >= 9 && exclude_first_col
        paths_sel(:,1) = false;
    end

    [dx,T,M,B1] = size(X_paths); %#ok<ASGLU>
    idx = find(paths_sel(:));
    if isempty(idx), error('No paths selected.'); end
    P = numel(idx);

    % Gather selected paths to 1×T×P
    Xp = zeros(1, T, P, 'like', X_paths);
    [mm, bb] = ind2sub([size(X_paths,3), size(X_paths,4)], idx);
    for k = 1:P
        Xp(:,:,k) = X_paths(:,:,mm(k), bb(k));
    end

    % Weights
    if nargin < 8 || isempty(weights)
        w = ones(P,1, 'like', Xp) / P;
    else
        w = weights(:) / sum(weights);
    end
    wrow = reshape(w, 1, P); %#ok<NASGU>

    % Shapes to T×P
    Xall = squeeze(Xp);            % T×P
    if isrow(Xall), Xall = Xall.'; end
    yrow = y(:).';

    % --- Observation term (dr)
    diff_yx = yrow.' - Xall;                 % T×P
    dr_each = -0.5./r + 0.5*(diff_yx.^2)/(r^2);  % T×P
    dr_k    = sum(dr_each, 1);               % 1×P
    dr      = dr_k * w;                      % scalar

    % --- Transition term (dtheta, dq)
    x_t   = Xall(2:end, :);                  % (T-1)×P
    x_tm1 = Xall(1:end-1, :);                % (T-1)×P
    res   = x_t - theta.*x_tm1;              % (T-1)×P
    dtheta_k = sum( (res .* x_tm1) / q, 1 ); % 1×P
    dq_k     = sum( -0.5./q + 0.5*(res.^2)/(q^2), 1 ); % 1×P
    dtheta   = dtheta_k * w;                 % scalar
    dq       = dq_k * w;                     % scalar

    % --- Initial term (dS0)
    x1 = Xall(1,:);                          % 1×P
    dS0_k = -0.5./S0 + 0.5*(x1.^2)/(S0^2);  % 1×P
    dS0    = dS0_k * w;                      % scalar

    out.avg_obs   = dr;
    out.avg_trans = [dtheta; dq];
    out.avg_init  = dS0;
    out.avg_total = [dr; dtheta; dq; dS0];
end


function trace = sa_pg_gauss_ssm(y,T,N,M,B,K,seed0, ...
                                 theta0,q0,r0,S0, ...
                                 Gamma,alpha,theta_max)
% SA stochastic approximation for Gaussian SSM via PG score.

% Unconstrained params
lq = log(q0); lr = log(r0);
theta = theta0;

trace.theta = zeros(K,1);
trace.q = zeros(K,1);
trace.r = zeros(K,1);
trace.score = zeros(3,K);
trace.step  = zeros(K,1);

for n=1:K
    t0 = tic;

    q = exp(lq); r = exp(lr);

    % --- Build PF/CPF functions ---
    in_pars.mu=0; in_pars.Sigma=S0;
    in_dist=@(p,N_,M_) reshape(p.mu+sqrt(p.Sigma)*randn(1,N_*M_),1,N_,M_);

    tr_pars.theta=theta; tr_pars.q=q; tr_pars.sig=sqrt(q);
    trans=@(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

    g_pars.R=r;
    g=@(yt,Xt,p,t) reshape(-0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R),1,size(Xt,2),size(Xt,3));

    trans_logpdf=@(x_next,X_prev,pars,t) -0.5*((x_next-theta*X_prev).^2)/q - 0.5*log(2*pi*q);

    % --- Run PG to approximate score ---
    out_pg=pgibbs_run_init_pfmean( ...
        y,T,N,M,B, ...
        in_dist,in_pars, ...
        trans,tr_pars, ...
        g,g_pars, ...
        seed0+1000*n, ...
        "cpf_parallel", ...
        "backward",trans_logpdf, ...
        false,false,false, ...
        N,"weighted");

    X_paths=out_pg.X_paths;
    S = score_gaussian_from_paths_vectorized(y,X_paths,theta,q,r,S0,[],[],true);
    g_vec = [S.avg_trans(1); S.avg_trans(2); S.avg_obs];  % [dtheta; dq; dr]

    % --- Convert to unconstrained grads
    grad_theta = g_vec(1);
    grad_lq    = q * g_vec(2);
    grad_lr    = r * g_vec(3);

    % --- Step sizes
    step_theta = Gamma(1)/(100+n)^(alpha+0.5);
    step_q     = Gamma(2)/(100+n)^(alpha+0.5);
    step_r     = Gamma(3)/(100+n)^(alpha+0.5);

    % --- Updates
    theta = theta + step_theta * grad_theta;
    lq    = lq    + step_q     * grad_lq;
    lr    = lr    + step_r     * grad_lr;

    % projection
    theta = max(min(theta, theta_max), -theta_max);

    % store
    trace.theta(n)=theta;
    trace.q(n)=exp(lq);
    trace.r(n)=exp(lr);
    trace.score(:,n)=g_vec;
    trace.step(n)=toc(t0);
end
end

function [unb_est, info] = pg_unbiased_score_gauss( ...
    y, T, N, M, ...
    in_dist_samp, in_pars, ...
    trans_dist_samp, trans_pars, ...
    g, g_pars, ...
    theta, q, r, S0, ...
    Bs, l_dist, seed0, ...
    cpf_choice, traj_mode, trans_logpdf, ...
    store_particles, store_ancestors, store_logw, ...
    N_init, init_mode)

% Unbiased estimator of [dℓ/dθ; dℓ/dq; dℓ/dr] using randomized PG levels
% for the 1D Gaussian SSM.

    % ---- basic checks ----
    Bs     = Bs(:).';
    l_dist = l_dist(:).';
    Lmax   = numel(Bs) - 1;
    if numel(l_dist) ~= Lmax+1
        error('Bs must have length Lmax+1 and l_dist must have same length.');
    end

    eLes = 0:Lmax;

    % ---- sample level l ~ l_dist ----
    rng(seed0);
    log_lw = log(l_dist);
    j      = draw_from_logw(log_lw);   % index in 1..Lmax+1
    l      = eLes(j);                  % actual level

    Bs_lvl = Bs(1:l+1);                % cumulative chain lengths for this level
    n_lvl  = numel(Bs_lvl);

    paths_sums = zeros(n_lvl, 3);
    total_cost = 0;                    % simple cost measure: sum of B_inc

    for n = 1:n_lvl
        if n > 1
            B_inc = Bs_lvl(n) - Bs_lvl(n-1);
        else
            B_inc = Bs_lvl(1);
        end

        total_cost = total_cost + B_inc;

        out_pg = pgibbs_run_init_pfmean( ...
            y, T, N, M, B_inc, ...
            in_dist_samp, in_pars, ...
            trans_dist_samp, trans_pars, ...
            g, g_pars, ...
            seed0 + 1000*n, ...
            cpf_choice, ...
            traj_mode, trans_logpdf, ...
            store_particles, store_ancestors, store_logw, ...
            N_init, init_mode);

        X_paths = out_pg.X_paths;

        S = score_gaussian_from_paths_vectorized( ...
                y, X_paths, theta, q, r, S0, ...
                [], [], true);  % use all paths except initializer

        g_vec = [S.avg_trans(1); S.avg_trans(2); S.avg_obs];  % [dθ; dq; dr]

        paths_sums(n,:) = B_inc * g_vec.';   % 1×3
    end

    if l > 0
        num = paths_sums(end,:) - sum(paths_sums(1:end-1,:), 1);
        den = Bs_lvl(end) * l_dist(j);
        unb_row = num / den;
    else
        num = paths_sums(1,:);
        den = Bs_lvl(1) * l_dist(j);
        unb_row = num / den;
    end

    unb_est = unb_row.';  % 3×1

    info.level      = l;
    info.level_idx  = j;
    info.Bs_used    = Bs_lvl;
    info.l_dist     = l_dist;
    info.paths_sums = paths_sums;
    info.seed0      = seed0;
    info.total_cost = total_cost;  % this is what we'll use later
end