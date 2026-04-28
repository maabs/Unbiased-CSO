function x = apply_givens_transpose_vec(x, omegas, Givset)
% Applies G' = Π_r G_r' (reverse order with transpose rotation).
x = x(:);
c = cos(omegas(:));
s = sin(omegas(:));
R = size(Givset,1);
for r = R:-1:1
    i = Givset(r,1); j = Givset(r,2);
    xi = x(i); xj = x(j);
    % transpose of [c -s; s c] is [c s; -s c]
    x(i) =  c(r)*xi + s(r)*xj;
    x(j) = -s(r)*xi + c(r)*xj;
end
end