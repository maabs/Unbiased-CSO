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


