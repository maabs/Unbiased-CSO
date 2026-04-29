function [m_f, P_f, m_s, P_s] = rts_smoother_1d(y, rho, q, r, H, m0, P0)
% Forward: Kalman filter (update at t, then predict to t+1)
T = size(y,2);
m_f = zeros(1,T); P_f = zeros(1,T);
m = m0; P = P0;
for t = 1:T
    % Update with y_t
    S = H*P*H' + r;                 % innovation variance
    K = (P*H') / S;                 % gain
    innov = y(1,t) - H*m;
    m = m + K*innov;
    P = (1 - K*H)*P;
    m_f(t) = m; P_f(t) = P;
    % Predict to t+1
    if t < T
        m = rho*m;
        P = rho*P*rho + q;
    end
end

% Backward: RTS smoothing
m_s = zeros(1,T); P_s = zeros(1,T);
m_s(T) = m_f(T); P_s(T) = P_f(T);
for t = T-1:-1:1
    % Predict stats from t to t+1 (using filtered at t)
    m_pred = rho * m_f(t);
    P_pred = rho * P_f(t) * rho + q;
    % Smoother gain
    C = (P_f(t) * rho) / P_pred;
    % Smoothed mean/var
    m_s(t) = m_f(t) + C * (m_s(t+1) - m_pred);
    P_s(t) = P_f(t) + C^2 * (P_s(t+1) - P_pred);
end
end

