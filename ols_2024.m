function results=ols_2024(y,x,robust)
% PURPOSE: least-squares regression 
%---------------------------------------------------
% USAGE: results = ols_2024(y,x)
% where: y = dependent variable vector    (nobs x 1)
%        x = independent variables matrix (nobs x nvar)
%---------------------------------------------------
% RETURNS: a structure
%        results.meth  = 'ols'
%        results.beta  = bhat     (nvar x 1)
%        results.yhat  = yhat     (nobs x 1)
%        results.resid = residuals (nobs x 1)
%        results.sige  = e'*e/(n-k)   scalar
%        results.rsqr  = rsquared     scalar
%        results.nobs  = nobs
%        results.nvar  = nvars
%        results.y     = y data vector (nobs x 1)
%        results.Sigma  = Sigma covariance matrix of OLS
%        results.Px    =NxN projection matrix 
%        results.Mx    =NxN residual maker matrix 
%        results.h = diagonal elements of the projection matrix
%        results.influence = u*h/(1-h)
%        results.s2 = variance of resid
%        results.Sigma = covariance matrix of beta (onder homoskedasticity)
%        results.tstat=Kx1 t-test for each coefficient
%        results.pvalue=Kx1 vector with pvalues for each coefficient
%        results.F_stat    =F-test for the existence of the regression 
%        results.p_value_F    =p_value F_test 
%        results.h = diagonal elements of the projection matrix
%        results.influence = u*h/(1-h)
%        results.s2 = variance of resid
%        results.Sigma = covariance matrix of beta (under homoskedasticity)
%        results.F_test_RESET= F-test for Reset test
%        results.p_val_RESET = p-value reset test
%        results.adjrsqr = Adjusted R2

if (nargin == 2);
    robust=0; % homoskedastic covariance matrix
elseif (nargin < 2);
    error('Wrong # of arguments to ols');
elseif (nargin > 3);
    error('Wrong # of arguments to ols');
end;
    

[nobs nvar] = size(x);
[nobs2 junk] = size(y);
    if (nobs ~= nobs2);
        error('x and y must have same # obs in ols');
    end;


results.meth = 'ols';
results.y = y;
results.nobs = nobs;
results.nvar = nvar;

if nobs < 10000
  [q r] = qr(x,0);
  xpxi = (r'*r)\eye(nvar);
else % use Cholesky for very large problems
  xpxi = (x'*x)\eye(nvar);
end;

results.beta = xpxi*(x'*y);
results.yhat = x*results.beta;
results.resid = y - results.yhat;
sigu = results.resid'*results.resid;
results.sige = sigu/(nobs-nvar);
ym = y - mean(y);
rsqr1 = sigu;
rsqr2 = ym'*ym;
results.rsqr = 1.0 - rsqr1/rsqr2; % r-squared
if robust==1;
D=diag(results.resid.^2);
results.Sigma=inv(x'*x)*(x'*D*x)*inv(x'*x);
else;
results.Sigma=results.sige*xpxi;
end;
sigb=sqrt(diag(results.Sigma));
results.bstd = sigb;
results.tstat = results.beta./sigb;
results.pvalue  = 2*(1-tcdf( abs(results.tstat), nobs-size(results.beta,1) ));
%% Comment this for speed
results.Px=x*xpxi*x';
results.Mx=eye(nobs)-x*xpxi*x';
results.h=diag(results.Px);
results.influence=results.h.*results.resid./(1-results.h);

%% F test- existsence of regression
K=size(results.beta,1);
% create matrix of restrictions
R=[zeros(K-1,1),eye(K-1)];
c=zeros(K-1,1);
results.F_stat=(R*results.beta-c)'*inv(R*results.Sigma*R')*(R*results.beta-c)/(K-1);
results.p_value_F=1-fcdf(results.F_stat,K-1,nobs-K);

%% Reset Test

y2=results.yhat.^2;
y3=results.yhat.^3;
x_tilde=[x,y2,y3];
beta_tilde=inv(x_tilde'*x_tilde)*x_tilde'*y;
epsilon_tilde=y-x_tilde*beta_tilde;
sigma2_tilde=(epsilon_tilde'*epsilon_tilde)/(nobs-nvar-2);
CV=sigma2_tilde*inv(x_tilde'*x_tilde);
R_tilde=[zeros(2,nvar),eye(2)];
c_tilde=[zeros(2,1)];
results.F_test_RESET=(R_tilde*beta_tilde-c_tilde)'*inv(R_tilde*CV*R_tilde')*(R_tilde*beta_tilde-c_tilde)/2;
results.p_val_RESET=1-fcdf(results.F_test_RESET,2,nobs-nvar-2);

%% Output on screen
parno   = (1:size(results.beta,1))';

Res     = [ parno results.beta results.bstd results.tstat results.pvalue];

 disp('OLS estimation');
 
fprintf('\n\n\n **********************************************************************\n');
if nobs-size(results.beta,1)<=0;
    fprintf('\nWarning\n')
    fprintf('Model contains more parameters than observations \n')
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n')
end
fprintf('Number of observations: %12.4f\n',nobs);
fprintf('Number of parameters    %12.4f\n',size(results.beta,1));
fprintf('Coeff. Determination (R^2)        %12.4f\n',results.rsqr);
fprintf('Existence of regression (p-value)    %12.4f\n',results.p_value_F);
fprintf('Ramsey RESET test (p-value)    %12.4f\n',results.p_val_RESET);
fprintf(' **********************************************************************\n');
fprintf('       parameter       beta        stderr    t-student      p-value\n');
fprintf('  %12.0f %12.4f  %12.4f %12.4f %12.4f\n', Res' );