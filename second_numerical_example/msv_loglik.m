function [loglik, v, gradOmega, gradHs] = msv_loglik(yt, omegas, hs, Givset)
%
%


N = length(yt); 
if nargout <=2
%    
    v = yt';

    for k = 1:size(Givset,1)
        i = Givset(k,1);
        j = Givset(k,2);
    
        c = cos(omegas(k));
        s = sin(omegas(k));
    
        vi = v(i);
        vj = v(j);
    
        v(i) = c*vi - s*vj;
        v(j) = s*vi + c*vj;
    end


    %for k=1:size(Givset,1)
    %   % OLD CODE -- SLOW
    %   %G = givensmat(omegas(k), N, Givset(k,1), Givset(k,2));
    %   %v = v*G;
    %
    %   % NEW CODE -- YOU NEVER NEED TO COMPUTE A GIVENS MATRIX!
    %   c = cos(omegas(k));
    %   s = sin(omegas(k));    
    %   tmp = v;
    %   v(Givset(k,1)) = c*tmp(Givset(k,1)) - s*tmp(Givset(k,2));
    %   v(Givset(k,2)) = s*tmp(Givset(k,1)) + c*tmp(Givset(k,2));
    %end
    
    v = v.*exp(-0.5*hs');
    loglik = - 0.5*N*log(2*pi) - 0.5*sum(hs) - 0.5*(v*v');
%
elseif nargout > 2
%    
    vGrad = zeros(size(Givset,1), 2);
    v = yt';
    c = cos(omegas);
    s = sin(omegas);  
    for k = 1:size(Givset,1)
        i = Givset(k,1);
        j = Givset(k,2);
    
        vi = v(i);
        vj = v(j);
    
        v(i) = c(k)*vi - s(k)*vj;
        v(j) = s(k)*vi + c(k)*vj;
    
        %vGrad(k, 1) = -s(k)*vi - c(k)*vj;
        %vGrad(k, 2) =  c(k)*vi - s(k)*vj;
        vGrad(k,1) = -v(j);
        vGrad(k,2) =  v(i);
    end
    %for k=1:size(Givset,1)
    %%    
    %  tmp = v;
    %  v(Givset(k,1)) = c(k)*tmp(Givset(k,1)) - s(k)*tmp(Givset(k,2));
    %  v(Givset(k,2)) = s(k)*tmp(Givset(k,1)) + c(k)*tmp(Givset(k,2));
    %  
    %  %vGrad(k, 1) = -s(k)*tmp(Givset(k,1)) - c(k)*tmp(Givset(k,2));
    %  %vGrad(k, 2) = c(k)*tmp(Givset(k,1)) - s(k)*tmp(Givset(k,2));
    %  vGrad(k,1) = - v(Givset(k,2));
    %  vGrad(k,2) = v(Givset(k,1));
    %%  
    %end
        
    v = v.*exp(-0.5*hs');
    keep = v; % keep it
    
    loglik = - 0.5*N*log(2*pi) - 0.5*sum(hs) - 0.5*(v*v');

    gradHs = - 0.5 + 0.5*(v.*v); 
    
    v = v.*exp(-0.5*hs');
    
    gradOmega = zeros(1, size(Givset,1)); 
    % backward
    for k = size(Givset,1):-1:1
        i = Givset(k,1);
        j = Givset(k,2);
    
        gradOmega(k) = vGrad(k,1)*v(i) + vGrad(k,2)*v(j);
    
        vi = v(i);
        vj = v(j);
    
        v(i) = c(k)*vi + s(k)*vj;
        v(j) = -s(k)*vi + c(k)*vj;
    end
    %for k=size(Givset,1):-1:1 
    %%     
    %  tmp = v;
    %  
    %  gradOmega(k) = vGrad(k,1)*v(Givset(k,1)) + vGrad(k,2)*v(Givset(k,2)); 
    %  
    %  v(Givset(k,1)) = c(k)*tmp(Givset(k,1)) + s(k)*tmp(Givset(k,2));
    %  v(Givset(k,2)) = -s(k)*tmp(Givset(k,1)) + c(k)*tmp(Givset(k,2));
    %%  
    %end
    %gradOmega = - gradOmega; 
    %v = keep; 
%    
end
    
    
    
