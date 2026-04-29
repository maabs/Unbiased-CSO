function [m_hist, P_hist] = kalman_1d(y, rho, q, r, H, m0, P0)
% y   : 1 × T
% rho : scalar state coefficient
% q   : process variance
% r   : observation variance
% H   : scalar observation matrix
% m0, P0 : prior mean/var for x_1
    T = size(y, 2);
    m = m0; P = P0;
    m_hist = zeros(1, T);
    P_hist = zeros(1, T);

    for t = 1:T
        % --- update with y_t ---
        S     = H*P*H' + r;              % innovation variance
        K     = (P*H') / S;              % Kalman gain
        innov = y(1, t) - H*m;           % innovation
        m     = m + K*innov;             % posterior mean
        P     = (1 - K*H)*P;             % posterior variance

        m_hist(t) = m;
        P_hist(t) = P;

        % --- predict to t+1 (skip after final step) ---
        if t < T
            m = rho*m;
            P = rho*P*rho + q;           % since scalar, rho' = rho
        end
    end
end

%%%
