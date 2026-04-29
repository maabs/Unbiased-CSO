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


