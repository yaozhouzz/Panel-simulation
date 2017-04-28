%% Data Generating Process
seed=100; rng(seed);%rng(n) is used to set the random seeds, n is a non-negative integer

N=100; T=20; K=1; L=1; nreps=1000; 

alpha_y=.5; beta=1-alpha_y; alpha_x=0.5; alpha=0.05*100; mew_lamda=0; 
corr_lamda=0.6; sigma2_lamda=1; ksi_x=1; ksi_y=1; %1/(1+mew_lamda^2);

%START OF SIMULATIONS%

mew_x=1*zeros(1,T); mew_y=1*zeros(1,T);
y=zeros(N,T); x=zeros(N,T); rank_M=zeros(nreps,1); var_y=zeros(nreps,T); 
stat1=zeros(nreps,1); prct1=zeros(nreps,1); stat2=zeros(nreps,1); 
prct2=zeros(nreps,1); statistic=zeros(nreps,1); pvalues=zeros(nreps,3);
% zeros means constuct a matrix N*T with all entries are zero.

for r=1:nreps    
    v=randn(N,T); e=randn(N,T);
    % randn means construct a N*T matrix with values drawn from N(0,1)
    lamda1=randn(N,1)+ones(N,1)*mew_lamda; 
    lamda1_noise=randn(N,1); 
    lamda1_x=corr_lamda*lamda1+sqrt(1-corr_lamda^2)*lamda1_noise+ones(N,1)*(mew_lamda);   
    % the lamda1_x here is lambda_i_*_1 in the model
    F1=1+randn(T,1)*1; F=F1; lamda=lamda1; lamda_x=lamda1_x;
        
%     x(:,1)=lamda_x*F(1,:)'+v(:,1);
%     y(:,1)=beta*x(:,1)+lamda*F(1,:)'+e(:,1);  
    
    x(:,1)=lamda_x*ksi_x+v(:,1);
    y(:,1)=lamda*ksi_y+e(:,1);  
    
    for t=2:T
    x(:,t)=alpha_x*x(:,t-1)+lamda_x*F(t,:)'+v(:,t);
    y(:,t)=alpha_y*y(:,t-1)+beta*x(:,t)+lamda*F(t,:)'+e(:,t);
    end
    
   if mod(r, 200) == 0 %return the reminder of r/200
        r
    else %this codes just check if the loop is finished.
    end

% *************************************
end

%% convert the wide-formate data to long-format
y_long=reshape(y',[],1);
x_long=reshape(x',[],1);

%% SUR
%this SUR has problems. Maybe what I do here is just GLS or Robust OLS. It
%is not FGLS
 x_sur=zeros(N*T,T);
 for i=1:N
    x_sur(1+(i-1)*T:i*T,i)=x_long(1+(i-1)*T:i*T);
 end
 beta_OLS=inv((x_sur'*x_sur))*x_sur'*y_long;
 error_OLS=y_long-x_sur*beta_OLS;
 error_reshape=reshape(error_OLS,[],N);
 var_cov=cov(error_reshape);
 var_cov_expand=kron(var_cov,eye(T));
 beta_SUR=inv((x_sur'*inv(var_cov_expand)*x_sur))*x_sur'*inv(var_cov_expand)*y_long;
 beta_SUR_estimate=mean(beta_SUR);


%% another SUR
%this SUR looks better. At least it is not available when N>T
if N<=T
error_SUR2=zeros(T,1);
for i=1:N
    [b,bint,r]=regress(y_long(1+(i-1)*T:i*T),x_long(1+(i-1)*T:i*T));
    error_SUR2=[error_SUR2,r];
end 
error_SUR2=error_SUR2(:,2:end);
var_cov2=cov(error_SUR2);
var_cov_expand2=kron(var_cov2,eye(T));
beta_SUR2=inv((x_sur'*inv(var_cov_expand2)*x_sur))*x_sur'*inv(var_cov_expand2)*y_long;
beta_SUR_estimate2=mean(beta_SUR2);
asym_var_estimation=inv((x_sur'*inv(var_cov_expand2)*x_sur));
%the variance is a matrix??
else
end

%% Principle component approach
error_PCA1=zeros(T,1);
for i=1:N
    [b,bint,r]=regress(y_long(1+(i-1)*T:i*T),x_long(1+(i-1)*T:i*T));
    error_PCA1=[error_PCA1,r];
end 
error_PCA1=error_PCA1(:,2:end)';
%transform it to N*T matrix because for each i, the principle component for
%each time is the same.
prin_component_matrix=pcacov(cov(error_PCA1));
principle1=prin_component_matrix(:,1);
principle2=prin_component_matrix(:,2);
beta_PCA=zeros(N,1);
error_PCA2=zeros(T,N);
for i=1:N
    regressor=[x_long(1+(i-1)*T:i*T),principle1,principle2];
    [b,bint,r]=regress(y_long(1+(i-1)*T:i*T),regressor);
    beta_PCA(i)=b(1);
    error_PCA2(:,i)=r;
end 
beta_PCA_estimation=mean(beta_PCA);
%this estimation should be OK, but why it is still so biased????





