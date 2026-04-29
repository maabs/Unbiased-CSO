function trace = sa_pg_gauss_ssm(y,T,N,M,B,K,seed0, ...
                                 theta0,q0,r0,S0, ...
                                 Gamma,alpha,theta_max)
% SA stochastic approximation for Gaussian SSM via PG score.

% Unconstrained params
lq = log(q0); lr = log(r0);
theta = theta0;

trace.theta = zeros(K,1);
trace.q = zeros(K,1);
trace.r = zeros(K,1);
trace.score = zeros(3,K);
trace.step  = zeros(K,1);

for n=1:K
    t0 = tic;

    q = exp(lq); r = exp(lr);

    % --- Build PF/CPF functions ---
    in_pars.mu=0; in_pars.Sigma=S0;
    in_dist=@(p,N_,M_) reshape(p.mu+sqrt(p.Sigma)*randn(1,N_*M_),1,N_,M_);

    tr_pars.theta=theta; tr_pars.q=q; tr_pars.sig=sqrt(q);
    trans=@(Xprev,p,t) p.theta*Xprev + p.sig*randn(size(Xprev),'like',Xprev);

    g_pars.R=r;
    g=@(yt,Xt,p,t) reshape(-0.5*((yt - Xt).^2)/p.R - 0.5*log(2*pi*p.R),1,size(Xt,2),size(Xt,3));

    trans_logpdf=@(x_next,X_prev,pars,t) -0.5*((x_next-theta*X_prev).^2)/q - 0.5*log(2*pi*q);

    % --- Run PG to approximate score ---
    out_pg=pgibbs_run_init_pfmean( ...
        y,T,N,M,B, ...
        in_dist,in_pars, ...
        trans,tr_pars, ...
        g,g_pars, ...
        seed0+1000*n, ...
        "cpf", ...
        "backward",trans_logpdf, ...
        false,false,false, ...
        N,"weighted");

    X_paths=out_pg.X_paths;
    S = score_gaussian_from_paths_vectorized(y,X_paths,theta,q,r,S0,[],[],true);
    g_vec = [S.avg_trans(1); S.avg_trans(2); S.avg_obs];  % [dtheta; dq; dr]

    % --- Convert to unconstrained grads
    grad_theta = g_vec(1);
    grad_lq    = q * g_vec(2);
    grad_lr    = r * g_vec(3);

    % --- Step sizes
    step_theta = Gamma(1)/(100+n)^(alpha+0.5);
    step_q     = Gamma(2)/(100+n)^(alpha+0.5);
    step_r     = Gamma(3)/(100+n)^(alpha+0.5);

    % --- Updates
    theta = theta + step_theta * grad_theta;
    lq    = lq    + step_q     * grad_lq;
    lr    = lr    + step_r     * grad_lr;

    % projection
    theta = max(min(theta, theta_max), -theta_max);

    % store
    trace.theta(n)=theta;
    trace.q(n)=exp(lq);
    trace.r(n)=exp(lr);
    trace.score(:,n)=g_vec;
    trace.step(n)=toc(t0);
end
end

