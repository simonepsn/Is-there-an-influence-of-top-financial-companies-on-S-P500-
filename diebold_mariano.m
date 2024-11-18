function [dm_stat, p_value] = diebold_mariano(y_true, pred1, pred2, h)
    if h < 0 || mod(h, 1) ~= 0
        error('Number of ACF lags must be a non-negative integer.');
    end

    e1 = y_true - pred1;
    e2 = y_true - pred2;
    d = e1 - e2;
    d_mean = mean(d);

    n = length(d);
    var_d = var(d);

    if var_d == 0
        fprintf('Variance of error differences is zero. No variability between model predictions.\n');
        dm_stat = NaN;
        p_value = NaN;
        return;
    end

    if h > 0
        lagged_covariances = xcov(d, h, 'biased');
        var_d = sum(lagged_covariances(h+1:2*h+1));
    end

    dm_stat = sqrt(n) * (d_mean / sqrt(var_d));

    p_value = 2 * (1 - normcdf(abs(dm_stat), 0, 1));
end

