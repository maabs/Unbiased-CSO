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
for m = 1:M
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






