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
