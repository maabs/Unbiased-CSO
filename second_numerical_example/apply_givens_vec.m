function v = apply_givens_vec(v, omegas, Givset, mode)
% mode = 'G' for v <- G*v
% mode = 'GT' for v <- G'*v
    c = cos(omegas(:));
    s = sin(omegas(:));
    R = size(Givset,1);

    if strcmp(mode,'G')
        % forward: same loop direction as constructing G = Gtmp*G (left-multiplying)
        for k = 1:R
            i = Givset(k,1); j = Givset(k,2);
            vi = v(i); vj = v(j);
            v(i) = c(k)*vi - s(k)*vj;
            v(j) = s(k)*vi + c(k)*vj;
        end
    elseif strcmp(mode,'GT')
        % transpose: apply inverse rotations in reverse order
        for k = R:-1:1
            i = Givset(k,1); j = Givset(k,2);
            vi = v(i); vj = v(j);
            % inverse of [c -s; s c] is [c s; -s c]
            v(i) = c(k)*vi + s(k)*vj;
            v(j) = -s(k)*vi + c(k)*vj;
        end
    else
        error('mode must be G or GT');
    end
end