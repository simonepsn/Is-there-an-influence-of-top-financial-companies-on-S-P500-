clc
clear

load Data_active
load Data_3factors

active = [BEN, BLK, BWG, BX, FHI, SEIC, VGT];
returns_active = diff(log(active));
returns_SPX = diff(log(SPX));
returns_VIX = diff(log(AdjClose));
factors = [MKTRF/100, SMB/100, HML/100];

% delete rows that don't match inbetween
factors = factors(1:2754, :);
returns_active = returns_active(1:2754, :);
returns_SPX = returns_SPX(1:2754, :);

N = size(returns_SPX);
iota = ones(N(1), 1);

y = returns_SPX;
X = [iota, returns_active];

% Performance of the investment in passive vs active context
% inv = 100;
% active_return_avg = mean(returns_active, 2);
% value_active = inv * cumprod(1 + active_return_avg);
% value_passive = inv * cumprod(1 + returns_SPX);

% Regression of all the actively managed companies against the benchmark
model1 = ols_2024(y, X, 1);
 
K = size(X, 2) - 1;
R2_1 = model1.rsqr;
adjusted_R2_1 = 1 - ((1 - R2_1) * (N - 1) / (N - K - 1));

% Regression of active portfolios against the benchmark and Fama-French
% factors

X2 = [iota, returns_active, factors(:,2), factors(:,3)];
model2 = ols_2024(y, X2, 1);
R2_2 = model2.rsqr;
K2 = size(X2, 2) - 1;
adjusted_R2_2 = 1 - ((1 - R2_2) * (N - 1) / (N - K2 - 1));

% In-sample and out-of-sample data split
X_in1 = X(1:2203, :);
y_in1 = y(1:2203, :);
X_out1 = X(2204:2754, :);
y_out1 = y(2204:2754, :);

model_in1 = ols_2024(y_in1, X_in1, 1);
y_fitted1 = X_out1 * model_in1.beta;

residuals_out1 = y_out1 - y_fitted1;
MSE1 = mean(residuals_out1 .^ 2);

% In-sample and out-of-sample for Model 2
X_in2 = X2(1:2203, :);
y_in2 = y(1:2203, :);
X_out2 = X2(2204:2754, :);
y_out2 = y(2204:2754, :);

model_in2 = ols_2024(y_in2, X_in2, 1);
y_fitted2 = X_out2 * model_in2.beta;


residuals_out2 = y_out2 - y_fitted2;
MSE2 = mean(residuals_out2 .^ 2);


% let's set up a random forest model 
% numTreesMax = 20;  
% oobErrorList = zeros(numTreesMax, 1);
% 
% for numTrees = 1:numTreesMax
%     model = TreeBagger(numTrees, X_in1, y_in1, 'Method', 'regression', 'OOBPrediction', 'On');
%     oobErrorList(numTrees) = oobError(model, 'Mode', 'ensemble');
% end


% the optimal number of trees is 16, hence

optimalNumTrees = 16;
randomForest = TreeBagger(optimalNumTrees, X_in1, y_in1, 'Method', 'regression', 'OOBPredictorImportance', 'on', 'OOBPrediction', 'on');

y_pred_rf = predict(randomForest, X_out1);

SSE_RF = sum((y_out1 - y_pred_rf).^2);
MSE_RF = SSE_RF / numel(y_out1);

% let's try a NN approach
X_train = X_in2';
Y_train = y_in2';
X_test = X_out2';
Y_test = y_out2';

net = fitnet([10, 8]);
net.trainParam.epochs = 1000; 
net.trainParam.showWindow = false;

bestMSE = inf;
bestNet = net;

maxEpochs = 1000;
maxFail = 50;
failCount = 0;

for i = 1:maxEpochs
    net = train(net, X_train, Y_train);
    Y_val_pred = net(X_test);
    currentMSE = perform(net, Y_test, Y_val_pred);
    
   
    if currentMSE < bestMSE
        bestMSE = currentMSE;
        bestNet = net;
        failCount = 0;
    else
        failCount = failCount + 1;
        if failCount >= maxFail
            disp(['Training stopped early at epoch ' num2str(i)]);
            break;
        end
    end

end

net = bestNet;

Y_pred_nn = bestNet(X_test);

MSE_nn = perform(bestNet, Y_test, Y_pred_nn);

% Let's run the Diebold-Mariano test 
[dm_stat2_NN, p_value_2_NN] = diebold_mariano(y_out2, y_fitted2, Y_val_pred, 0);
[dm_stat2_RF, p_value_2_RF] = diebold_mariano(y_out2, y_fitted2, y_pred_rf, 0);
[dm_statNN_RF, p_value_NN_RF] = diebold_mariano(y_out2, Y_val_pred, y_pred_rf, 0);

% Comparison between NN and reference model
fprintf('\nComparison: Neural Network vs. Reference Model\n');
fprintf('P-value: %.4f\n', p_value_2_NN);
if p_value_2_NN < 0.05
    disp('Result: The difference in predictive accuracy is statistically significant.');
else
    disp('Result: No significant difference in predictive accuracy.');
end

% Comparison between RF and reference model
fprintf('\nComparison: Random Forest vs. Reference Model\n');
fprintf('P-value: %.4f\n', p_value_2_RF);
if p_value_2_RF < 0.05
    disp('Result: The difference in predictive accuracy is statistically significant.');
else
    disp('Result: No significant difference in predictive accuracy.');
end

% Comparison between NN and RF
fprintf('\nComparison: Neural Network vs. Random Forest\n');
fprintf('P-value: %.4f\n', p_value_NN_RF);
if p_value_NN_RF < 0.05
    disp('Result: The difference in predictive accuracy is statistically significant.');
else
    disp('Result: No significant difference in predictive accuracy.');
end


% Compute forecast errors for each model
forecast_errors_nn = Y_test - Y_pred_nn;
forecast_errors_rf = y_out1 - y_pred_rf;
forecast_errors_model2 = y_out2 - y_fitted2;

% Here, we use a common rule of thumb for the bandwidth.
bandwidth = floor(4 * (size(X_train, 2) / 100)^(2/9));


% Let's apply the fluctuation test from Giacomini (2010)

[F_stat_nn, cv_nn] = fluctuationTest(forecast_errors_nn, bandwidth, numel(forecast_errors_nn));

[F_stat_rf, cv_rf] = fluctuationTest(forecast_errors_rf, bandwidth, numel(forecast_errors_rf));

[F_stat_model2, cv_model2] = fluctuationTest(forecast_errors_model2, bandwidth, numel(forecast_errors_model2));

disp(['Fluctuation test statistic for NN model: ', num2str(F_stat_nn)]);
disp(['Fluctuation test statistic for RF model: ', num2str(F_stat_rf)]);
disp(['Fluctuation test statistic for OLS model2: ', num2str(F_stat_model2)]);

MSE_values = [MSE1, MSE2, MSE_RF, MSE_nn];

% The accuracy of the forecasts does not exhibit significant changes of 
% variance as we move through the time series

% % figures
% 
% figure;
% plot(value_passive, 'LineWidth', 2, 'DisplayName', 'Passive (SPX)');
% hold on;
% plot(value_active, 'LineWidth', 2, 'DisplayName', 'Active Portfolio');
% title('Investment Performance: Active vs Passive');
% xlabel('Time');
% ylabel('Investment Value');
% legend('Location', 'Best');
% grid on;
% figure;
% plot(1:numTreesMax, oobErrorList, '-o');
% xlabel('Number of Trees');
% ylabel('OOB Error');
% title('OOB Error vs. Number of Trees');
% grid on;
% 
% [optimalError, optimalNumTrees] = min(oobErrorList);
% fprintf('Optimal number of trees: %d\n', optimalNumTrees);
%
% % For Random Forest Model
% figure;
% plot(y_out1, 'k', 'LineWidth', 2, 'DisplayName', 'Actual');
% hold on;
% plot(y_pred_rf, ':r', 'LineWidth', 2, 'DisplayName', 'RF Prediction (Optimal)');
% legend('Location', 'best');
% title('Out-of-Sample Predictions: Random Forest (Optimal)');
% xlabel('Observation');
% ylabel('Values');
% grid on;
% hold off;
% 
% % Plotting the results
% 
% % For Model 1
% figure;
% plot(y_out1, 'k', 'LineWidth', 2, 'DisplayName', 'Actual');
% hold on;
% plot(y_fitted1, '--b', 'LineWidth', 2, 'DisplayName', 'Linear Prediction (Model 1)');
% legend('Location', 'best');
% title('Out-of-Sample Predictions: Model 1');
% xlabel('Observation');
% ylabel('Values');
% grid on;
% hold off;
% 
% % For Model 2
% figure;
% plot(y_out2, 'k', 'LineWidth', 2, 'DisplayName', 'Actual');
% hold on;
% plot(y_fitted2, '--g', 'LineWidth', 2, 'DisplayName', 'Linear Prediction (Model 2)');
% legend('Location', 'best');
% title('Out-of-Sample Predictions: Model 2');
% xlabel('Observation');
% ylabel('Values');
% grid on;
% hold off;
% 
% % Plotting the results for the neural network model
% figure;
% plot(Y_test', 'k', 'LineWidth', 2, 'DisplayName', 'Actual');
% hold on;
% plot(Y_pred_nn', ':r', 'LineWidth', 2, 'DisplayName', 'NN Prediction');
% legend('Location', 'best');
% title('Out-of-Sample Predictions: Neural Network');
% xlabel('Observation');
% ylabel('Values');
% grid on;
% hold off;
% 
% % Combined Graph for All Models and Predictions
% figure;
% plot(y_out1, 'k', 'LineWidth', 2, 'DisplayName', 'Actual');
% hold on;
% plot(y_fitted1, '--b', 'LineWidth', 2, 'DisplayName', 'Linear Prediction (Model 1)');
% plot(y_fitted2, '--g', 'LineWidth', 2, 'DisplayName', 'Linear Prediction (Model 2)');
% plot(y_pred_rf, '--p', 'LineWidth', 2, 'DisplayName', 'RF Prediction');
% plot(Y_pred_nn', ':r', 'LineWidth', 2, 'DisplayName', 'NN Prediction');
% title('Comparison of Out-of-Sample Predictions');
% xlabel('Observation');
% ylabel('Values');
% legend('Location', 'best');
% grid on;
% hold off
% model_labels = {'OLS Model 1', 'OLS Model 2', 'Random Forest', 'Neural Network'};
% 
% bar(MSE_values);
% set(gca, 'XTickLabel', model_labels);
% xlabel('Models');
% ylabel('Mean Squared Error (MSE)');
% title('Comparison of Mean Squared Error (MSE) for Different Models');
% 
% p_values = [p_value_2_NN, p_value_2_RF, p_value_NN_RF];
% test_stats = [dm_stat2_NN, dm_stat2_RF, dm_statNN_RF];
% labels = {'NN vs. Ref', 'RF vs. Ref', 'NN vs. RF'};
% figure;
% bar(p_values);
% set(gca, 'XTickLabel', labels);
% ylabel('P-value');
% title('Diebold-Mariano Test P-Values');
% ylim([0 1]);
% hold on;
% line(xlim(), [0.05 0.05], 'Color', 'red', 'LineStyle', '--');
% text(1, 0.06, 'Significance Threshold (0.05)', 'Color', 'red');
% for i = 1:length(p_values)
%     text(i, p_values(i) + 0.02, sprintf('%.4f', p_values(i)), 'HorizontalAlignment', 'center');
% end
% figure;
% bar(test_stats);
% set(gca, 'XTickLabel', labels);
% ylabel('Test Statistic');
% title('Giacomini-Rossi Test Statistics');
% for i = 1:length(test_stats)
%     text(i, test_stats(i) + 0.02, sprintf('%.4f', test_stats(i)), 'HorizontalAlignment', 'center');
% end
% Test statistics for each model
% F_stats = [F_stat_nn, F_stat_rf, F_stat_model2];
% critical_values = [cv_nn, cv_rf, cv_model2];
% model_labels = {'Neural Network', 'Random Forest', 'OLS Model'};
% figure;
% bar(F_stats);
% hold on;
% 
% for i = 1:length(critical_values)
%     line([i-0.2, i+0.2], [critical_values(i) critical_values(i)], 'Color', 'red', 'LineWidth', 2);
% end
% 
% set(gca, 'XTick', 1:length(model_labels), 'XTickLabel', model_labels);
% ylabel('Fluctuation Test Statistic');
% title('Comparison of Fluctuation Test Statistics Across Models');
% for i = 1:length(critical_values)
%     text(i, critical_values(i) + 0.1, sprintf('CV=%.2f', critical_values(i)), ...
%         'HorizontalAlignment', 'center', 'Color', 'red');
% end
% for i = 1:length(F_stats)
%     text(i, F_stats(i) - 0.1, sprintf('%.2f', F_stats(i)), ...
%         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'Color', 'blue');
% end
% 
% hold off; 