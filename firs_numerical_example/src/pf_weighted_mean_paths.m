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

