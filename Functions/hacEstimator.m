function sigma_hat_squared = hacEstimator(forecast_errors, bandwidth)
   
    T = length(forecast_errors);
    sigma_hat_squared = var(forecast_errors, 1);
    
    for lag = 1:bandwidth
        weight = 1 - (lag / (bandwidth + 1));
        
        forecast_errors_lag = forecast_errors((lag+1):end);
        forecast_errors_no_lag = forecast_errors(1:(end-lag));
        gamma = cov(forecast_errors_lag, forecast_errors_no_lag);
        
        sigma_hat_squared = sigma_hat_squared + 2 * weight * gamma(1,2);
    end
end
