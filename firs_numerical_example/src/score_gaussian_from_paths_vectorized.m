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
    dS0    = dS0_k * w;                     % scalar

    out.avg_obs   = dr;
    out.avg_trans = [dtheta; dq];
    out.avg_init  = dS0;
    out.avg_total = [dr; dtheta; dq; dS0];
end


