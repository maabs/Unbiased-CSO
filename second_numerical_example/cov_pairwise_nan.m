function S = cov_pairwise_nan(Y)
% Y: N x T, with NaNs
% Pairwise covariance using overlapping observations per pair.
[N,T] = size(Y);
S = zeros(N,N);

for i=1:N
    yi = Y(i,:);
    for j=i:N
        yj = Y(j,:);
        ind = ~isnan(yi) & ~isnan(yj);
        x = yi(ind);
        z = yj(ind);
        if numel(x) < 2
            cij = NaN;
        else
            x = x - mean(x);
            z = z - mean(z);
            cij = (x*z')/(numel(x)-1);
        end
        S(i,j) = cij;
        S(j,i) = cij;
    end
end

% Make SPD-ish (small jitter + sym)
S = (S+S')/2;
S = S + 1e-10*eye(N);
end