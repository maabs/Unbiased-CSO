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
