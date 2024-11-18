function [F_stat, critical_value] = fluctuationTest(forecast_errors, bandwidth, T)

    forecast_errors = forecast_errors(:);
    sigma_hat_squared = hacEstimator(forecast_errors, bandwidth);
    
    cumulative_sum = zeros(T, 1);
    
    for t = 1:T
        cumulative_sum(t) = sum(forecast_errors(1:t).^2);
    end
   
    F_stat = max(abs(cumulative_sum - mean(cumulative_sum))) / sqrt(sigma_hat_squared);
    
    critical_value = 1.96;
    
    if F_stat > critical_value
        disp('Reject the null hypothesis: Evidence of fluctuation');
    else
        disp('Fail to reject the null hypothesis: No evidence of fluctuation');
    end
end
