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
