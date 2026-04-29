function [score, parts] = score_gaussian_ssm(y, theta, q, r, S0)
% SCORE_GAUSSIAN_SSM  Gradient (score) of the log-likelihood for a 1D LGSSM.
% Model:
%   X1 ~ N(0, S0)
%   Xt | X_{t-1} ~ N(theta * X_{t-1}, q)
%   Yt | Xt ~ N(Xt, r)
%
% Inputs
%   y      : 1×T observations
%   theta  : scalar (state coefficient)
%   q      : scalar > 0 (state var)
%   r      : scalar > 0 (obs var)
%   S0     : scalar > 0 (initial var)
%
% Outputs
%   score  : struct with fields dtheta, dq, dr, dS0
%   parts  : struct with useful internals (filtered, smoothed, cross-cov, etc.)

    y  = y(:)';                 % 1×T
    T  = size(y,2);
    eps_small = 1e-12;

    % ---------- Forward: Kalman filter (update at t, then predict to t+1)
    m_f = zeros(1,T);  P_f = zeros(1,T);
    m_pred = zeros(1,T);  P_pred = zeros(1,T);   % store P_{t|t-1} (shifted)
    % prior for t=1 (before seeing y1)
    m = 0;
    P = S0;
    for t = 1:T
        % innovation update at t
        S_t = P + r;                          % 1×1
        K   = P / S_t;
        innov = y(t) - m;
        m = m + K * innov;
        P = (1 - K) * P;
        m_f(t) = m;  P_f(t) = P;

        % prediction to t+1
        if t < T
            m_pred(t+1) = theta * m;
            P_pred(t+1) = theta^2 * P + q;
            m = m_pred(t+1);
            P = P_pred(t+1);
        end
    end

    % ---------- Backward: RTS smoother + lag-one covariance
    m_s = zeros(1,T);  P_s = zeros(1,T);
    J   = zeros(1,T-1);           % smoother gains J_t for t=1..T-1
    C_lag = zeros(1,T);           % Cov(X_t, X_{t-1} | y), defined for t>=2

    m_s(T) = m_f(T);  P_s(T) = P_f(T);
    for t = T-1:-1:1
        % P_pred(t+1) is prediction variance from t to t+1
        Pp = max(P_pred(t+1), eps_small);
        J(t) = (P_f(t) * theta) / Pp;

        % smooth
        m_s(t) = m_f(t) + J(t) * (m_s(t+1) - theta*m_f(t));
        P_s(t) = P_f(t) + J(t)^2 * (P_s(t+1) - Pp);

        % lag-one covariance: Cov(X_t, X_{t-1} | y), here for index (t) vs (t-1)
        % Use Σ_{t-1,t}^s = J(t-1) * Σ_t^s  ⇒ Cov(X_t, X_{t-1}) = J(t-1) * P_s(t)
        % We'll fill it after loop for t>=2 using J(t-1)
    end
    for t = 2:T
        C_lag(t) = J(t-1) * P_s(t);   % Cov(X_t, X_{t-1} | y)
    end

    % ---------- Expectations needed for the Fisher score
    % E[X_t] = m_s(t);  Var[X_t] = P_s(t);
    % E[X_t^2] = P_s(t) + m_s(t)^2
    EX2  = P_s + m_s.^2;
    % E[X_t X_{t-1}] = Cov + mean product
    EXXt = C_lag + m_s .* [0, m_s(1:end-1)];   % first entry unused (t=1)

    % ---------- Score components
    % (1) wrt theta:  sum_{t=2..T} (1/q) E[(X_t - theta X_{t-1}) X_{t-1}]
    %     = (1/q) sum_{t=2..T} (E[X_t X_{t-1}] - theta E[X_{t-1}^2])
    E_XtXm1   = EXXt(2:end);
    E_Xm1sq   = EX2(1:end-1);
    dtheta = (1/max(q,eps_small)) * sum( E_XtXm1 - theta * E_Xm1sq );

    % (2) wrt q:  -(T-1)/(2q) + (1/(2q^2)) sum_{t=2..T} E[(X_t - theta X_{t-1})^2]
    % E[(X_t - theta X_{t-1})^2] = E[X_t^2] - 2theta E[X_t X_{t-1}] + theta^2 E[X_{t-1}^2]
    E_res2 = EX2(2:end) - 2*theta*E_XtXm1 + theta^2 * E_Xm1sq;
    dq = -(T-1)/(2*max(q,eps_small)) + 0.5 * sum(E_res2) / max(q,eps_small)^2;

    % (3) wrt r:  -T/(2r) + (1/(2r^2)) sum_t E[(Y_t - X_t)^2]
    % E[(Y_t - X_t)^2] = (y - m_s).^2 + P_s
    E_meas2 = (y - m_s).^2 + P_s;
    dr = -T/(2*max(r,eps_small)) + 0.5 * sum(E_meas2) / max(r,eps_small)^2;

    % (4) wrt S0 (initial variance):  -1/(2S0) + (1/(2S0^2)) E[X_1^2]
    dS0 = -1/(2*max(S0,eps_small)) + 0.5 * EX2(1) / max(S0,eps_small)^2;

    % ---------- Package
    score = struct('dtheta', dtheta, 'dq', dq, 'dr', dr, 'dS0', dS0);

    if nargout > 1
        parts = struct();
        parts.m_f = m_f; parts.P_f = P_f;
        parts.m_s = m_s; parts.P_s = P_s;
        parts.J = J; parts.P_pred = P_pred;
        parts.C_lag = C_lag;            % Cov(X_t, X_{t-1} | y), t>=2
        parts.EX2 = EX2; parts.EXXt = EXXt;
        parts.E_res2 = E_res2; parts.E_meas2 = E_meas2;
    end
end
