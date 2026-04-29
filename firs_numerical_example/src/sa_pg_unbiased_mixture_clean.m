function trace = sa_pg_unbiased_mixture_clean( ...
    theta0, q0, r0, ...               % initial physical parameters
    K_SA, Gamma, alpha, n0, ...       % SA steps & schedule
    m_mix, S, ...                     % mixture parameter & #unb draws per iter
    unb_gauss, unb_tstud, ...         % @(theta,q,r,seed)->[dθ; dq; dr]
    seed0)

    % Allocate traces in (θ, log q, log r)
    theta_trace = zeros(K_SA+1,1);
    lq_trace    = zeros(K_SA+1,1);
    lr_trace    = zeros(K_SA+1,1);
    grad_trace  = zeros(K_SA,3);
    gamma_trace = zeros(K_SA,3);
    family_trace= zeros(K_SA,1); % 1=Gaussian,2=t-Student

    % Init in transformed domain
    theta_trace(1) = theta0;
    lq_trace(1)    = log(q0);
    lr_trace(1)    = log(r0);

    Gamma = Gamma(:);   % 3×1
    rng(seed0,"twister");

    p_gauss = m_mix/(m_mix+1);

    for n = 1:K_SA

        % Current SA state
        theta_n = theta_trace(n);
        lq_n    = lq_trace(n);
        lr_n    = lr_trace(n);

        % Physical params
        q_n = exp(lq_n);
        r_n = exp(lr_n);

        % Step sizes
        gamma_n = Gamma ./ ( (n0 + n)^(alpha + 0.5) );
        gamma_trace(n,:) = gamma_n.';

        % Choose family
        if rand < p_gauss
            fam = 1;  % Gaussian
        else
            fam = 2;  % t-Stud
        end
        family_trace(n) = fam;

        % Collect S unbiased draws
        G_phys = zeros(3,S);   % in (θ,q,r)
        for s = 1:S
            seed_s = seed0 + 100000*n + 1000*s;
            if fam == 1
                G_phys(:,s) = unb_gauss(theta_n, q_n, r_n, seed_s);
            else
                G_phys(:,s) = unb_tstud(theta_n, q_n, r_n, seed_s);
            end
        end

        % Average & transform to (θ, log q, log r)
        U_raw = mean(G_phys,2);   % [dθ; dq; dr]

        g_hat = zeros(3,1);
        g_hat(1) = U_raw(1);
        g_hat(2) = U_raw(2) * q_n;  % d/d(log q)
        g_hat(3) = U_raw(3) * r_n;  % d/d(log r)

        grad_trace(n,:) = g_hat.';

        % SA update in transformed domain
        theta_next = theta_n + gamma_n(1)*g_hat(1);
        lq_next    = lq_n    + gamma_n(2)*g_hat(2);
        lr_next    = lr_n    + gamma_n(3)*g_hat(3);

        theta_trace(n+1) = theta_next;
        lq_trace(n+1)    = lq_next;
        lr_trace(n+1)    = lr_next;
    end

    % Pack output, both transformed and physical
    trace.theta  = theta_trace;
    trace.q      = exp(lq_trace);
    trace.r      = exp(lr_trace);
    trace.lq     = lq_trace;
    trace.lr     = lr_trace;
    trace.grad   = grad_trace;
    trace.gamma  = gamma_trace;
    trace.family = family_trace;
    trace.S      = S;
    trace.m_mix  = m_mix;
end


