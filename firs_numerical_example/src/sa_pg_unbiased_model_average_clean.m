function trace = sa_pg_unbiased_model_average_clean( ...
    theta0, q0, r0, ...               % initial physical parameters
    K_SA, Gamma, alpha, n0, ...       % SA steps & schedule
    m_star, S, ...                    % model space size & #unb draws per iter
    unb_gauss, unb_tstud, ...         % gauss: @(theta,q,r,seed)
    seed0)                                 % tstud: @(theta,q,r,nu,seed)
    

    % Allocate traces in (theta, log q, log r)
    theta_trace = zeros(K_SA+1,1);
    lq_trace    = zeros(K_SA+1,1);
    lr_trace    = zeros(K_SA+1,1);
    grad_trace  = zeros(K_SA,3);
    gamma_trace = zeros(K_SA,3);

    % Model trace:
    % M = 1,...,m_star-1 means Student-t with df M
    % M = m_star means Gaussian
    model_trace = zeros(K_SA,1);

    % Init in transformed domain
    theta_trace(1) = theta0;
    lq_trace(1)    = log(q0);
    lr_trace(1)    = log(r0);

    Gamma = Gamma(:);   % 3 x 1
    rng(seed0,"twister");

    for n = 1:K_SA

        % Current SA state
        theta_n = theta_trace(n);
        lq_n    = lq_trace(n);
        lr_n    = lr_trace(n);

        % Physical parameters
        q_n = exp(lq_n);
        r_n = exp(lr_n);

        % Step sizes
        gamma_n = Gamma ./ ((n0 + n)^(alpha + 0.5));
        gamma_trace(n,:) = gamma_n.';

        % Sample model index uniformly from {1,...,m_star}
        M = randi(m_star);
        model_trace(n) = M;

        % Collect S unbiased draws
        G_phys = zeros(3,S);   % in (theta,q,r)

        for s = 1:S
            seed_s = seed0 + 100000*n + 1000*s;

            if M == m_star
                % Gaussian observation model
                G_phys(:,s) = unb_gauss(theta_n, q_n, r_n, seed_s);
            else
                % Student-t observation model with df = M
                nu = M;
                G_phys(:,s) = unb_tstud(theta_n, q_n, r_n, nu, seed_s);
            end
        end

        % Average score in physical parameters
        U_raw = mean(G_phys,2);   % [dtheta; dq; dr]

        % Transform score to (theta, log q, log r)
        g_hat = zeros(3,1);
        g_hat(1) = U_raw(1);
        g_hat(2) = U_raw(2) * q_n;
        g_hat(3) = U_raw(3) * r_n;

        grad_trace(n,:) = g_hat.';

        % SA update in transformed domain
        theta_trace(n+1) = theta_n + gamma_n(1)*g_hat(1);
        lq_trace(n+1)    = lq_n    + gamma_n(2)*g_hat(2);
        lr_trace(n+1)    = lr_n    + gamma_n(3)*g_hat(3);
    end

    % Pack output
    trace.theta = theta_trace;
    trace.q     = exp(lq_trace);
    trace.r     = exp(lr_trace);
    trace.lq    = lq_trace;
    trace.lr    = lr_trace;
    trace.grad  = grad_trace;
    trace.gamma = gamma_trace;

    trace.model = model_trace;
    trace.S     = S;
    trace.m_star = m_star;
end