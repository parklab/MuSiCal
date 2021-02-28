function [W, H, lossfun, t, reconstructionerrors, volumes, linesearchsteps, gammas, lambda] = betaNMF_logdetKL(V,W0,H0,K,MAXITER,lambda_tilde,delta,gamma,alpha)

% Copied and modified from minvol-KL-NMF v1.0 by Daniel Ben-Isvy
% Reference: Valentin Leplat, Nicolas Gillis and Man Shun Ang, "Blind Audio Source Separation with Minimum-Volume Beta-Divergence NMF", 2019, preprint

% This function tackles the NMF problem with volume constraints on the
% factor W using the KL divergence:
%
% min_{W,H >= 0} D_KL(V | WH) + lambda * logdet(W^TW + delta I),
%
% where the paramters (lambda, delta) are specified via inputs.
% Note that lambda is specified via lambda_tilde: lambda will
% be equal to lambda_tilde * D_KL(V | WH) / logdet(W^TW + delta I) in
% order to balance the two terms properly in the objective, where (W,H) is
% the initialization.

%INPUTS:
%       V:                 Spectrogram of x (input audio signal)
%       K:         Number of sources (determined prior to the
%       factorization)
%       MAXITER:   Maximum number of iterations for updates of W and H
%       lambda_tilde:   Relative Weight for Volume minimization
%       delta:     Parameter within the logdet function
%       gamma:     Initial value for line search
%       alpha:     Parameter of the MU to control the sparsity of H


%OUTPUTS:
%       W: Matrix dictionnaries
%       H: Matrix activations
%       lossfun: value of the loss function f(W,H) after MAXITER iterations
%       t: time in seconds to return a solution

% % Setting additional parameters
beta=1;
init=1;

% % Algorithm Beta-NMF with logdet(W'W+delta*I) regularization
% disp(' ->min-vol KL-NMF algorithm')
tic
F = size(V,1);
T = size(V,2);
% addpath('./minvol_utils');

% %  Initialization for W and H

% Random initialization

% disp(' ->Random Initialization for W and H')
%rand('seed',0)
%W = 1+rand(F, K);
%H = 1+rand(K, T);
W = W0;
H = H0;


% % Initialization for loop parameters
traceSave=zeros(MAXITER,1);
logdetSave=zeros(MAXITER+1,1);
condNumberSave=zeros(MAXITER,1);
Y = inv(W'*W + delta*eye(K));

% % array to save the value of the loss function
lossfunsave = zeros(MAXITER+1,1);
linesearchsteps = zeros(MAXITER, 1);
gammas = zeros(MAXITER+1, 1);
reconstructionerrors = zeros(MAXITER+1, 1);

% % initialization for lambda
lambda=lambda_tilde*betaDiv(V+eps,W*H+eps,beta)/abs(log10(det(W'*W+delta*eye(K))));
if(init==0)
    fprintf(' ->The initial value for betadivergence is %0.2f \n', betaDiv(V+eps,W*H+eps,beta));
    fprintf(' ->The value for the penalty weight is %0.2f \n', lambda);
    fprintf(' ->The initial value for penalty term is %0.2f \n', lambda * log10(det(W'*W+delta*eye(K))));
    fprintf(' ->The initial ratio of terms is %0.2f \n', lambda * log10(det(W'*W+delta*eye(K)))/betaDiv(V+eps,W*H+eps,beta));
end

% Others parameters/variables
ONES = ones(F,T);
if(gamma ~= -1)
    W_prev=W;
end
if(init==0)
    subplot(2,1,1);
    ani1=animatedline('Color','r','Marker','o');
    title(['Evolution of objective function - $\beta$ = ' num2str(beta)],'FontSize',12, 'Interpreter','latex')
    xlabel('iteration','FontSize',12, 'Interpreter','latex')
    subplot(2,1,2)
    ani2=animatedline('Color','k','Marker','o');
    title('$\gamma$ evolution','FontSize',12, 'Interpreter','latex')
    xlabel('iteration','FontSize',12, 'Interpreter','latex')
end


logdetSave(1)=log10(det(W'*W+delta*eye(K)));
lossfunsave(1) = betaDiv(V+eps,W*H+eps,beta) + lambda * log10(det(W'*W+delta*eye(K)));
reconstructionerrors(1) = betaDiv(V+eps,W*H+eps,beta);
gammas(1)=gamma


% % Optimization loop
for iter=1:MAXITER

     % % update matrix  H Coefficients ("activations")
    H = (H .* (W'*(((W*H).^(beta-2)).*V))./(W'*(W*H).^(beta-1)+eps)).^(1+alpha);
    H = max(H,eps);

    %  % update maxtrix W ("dictionaries")
    Y_plus=max(0,Y);
    Y_minus=max(0,-Y);
    Wup = W .*(((ONES*H'-4*lambda*W*Y_minus).^2+8*lambda*(W*(Y_plus+Y_minus)).*((V./(W*H+eps))*H')).^(1/2)-ONES*H'+4*lambda*W*Y_minus)./(4*lambda*W*(Y_plus+Y_minus)+eps);
    min(min(Wup));
    Wup = max(Wup,eps);

    if(gamma ~= -1 && iter>5)
        W = (1-gamma)*W_prev + gamma*Wup;
        W = max(SimplexProjW(W),eps);
        k=0;
        while ((betaDiv(V+eps,W_prev*H+eps,beta)+lambda*log10(det(W_prev'*W_prev+delta*eye(K))))< (betaDiv(V+eps,W*H+eps,beta)+lambda*log10(det(W'*W+delta*eye(K)))) && gamma>1e-16)
            gamma=gamma*0.8;
            W = (1-gamma)*W_prev + gamma*Wup;
            W = max(SimplexProjW(W),eps);
            k=k+1;
        end
        W_prev=W;
    else
        k=0;
        W=max(SimplexProjW(Wup),eps);

    end

    % % Compute the loss function = Beta-Divergence + Penalty term
    lossfunsave(iter+1) = betaDiv(V+eps,W*H+eps,beta) + lambda * log10(det(W'*W+delta*eye(K)));
    traceSave(iter)=trace(Y*(W'*W));
    logdetSave(iter+1)=log10(det(W'*W+delta*eye(K)));
    condNumberSave(iter)=cond(W'*W+delta*eye(K));
    reconstructionerrors(iter+1) = betaDiv(V+eps,W*H+eps,beta);
    linesearchsteps(iter) = k;

    % % Update Y
    Y = inv(W'*W + delta*eye(K));

    % Drawing
    if(init==0)
        if(delta<1)
            addpoints(ani1,iter,lossfunsave(iter));
        else
            addpoints(ani1,iter,log10(lossfunsave(iter))) ;
        end
        addpoints(ani2,iter,gamma) ;
        drawnow
    end


    % % Update gamma
    if(gamma ~= -1)
        gamma=min(gamma*2,1);
    end
    gammas(iter+1) = gamma;

end

if(init==0)
    figure2=figure;
    plot((1:length(traceSave)),traceSave,1:length(logdetSave),logdetSave,1:length(logdetSave),condNumberSave);
    title('Evolution of logdet(Wt*W+delta*I) and Trace-UpperBound','FontSize',12, 'Interpreter','latex');
    xlabel('Iteration [-] ','FontSize',12, 'Interpreter','latex'), ylabel('[-] ','FontSize',12, 'Interpreter','latex')
    fprintf('\n');
    legend(sprintf('Trace-UpperBound'),sprintf('logdet(Wt*W+delta*I)'),sprintf('cond(Wt*W+delta*I)'))
    saveas(figure2,'./Graphs/logdet_trace_condNumber.png')
end


% % Save lossfun
lossfun=lossfunsave;
volumes=logdetSave;
% % Final Value for loss function
if(init==0)
    fprintf(' ->The final value for betadivergence is %0.2f \n', betaDiv(V+eps,W*H+eps,beta));
    fprintf(' ->The final value for penalty term is %0.2f \n', lambda * log10(det(W'*W+delta*eye(K))));
    fprintf(' ->The ratio of terms is %0.2f \n', lambda * log10(det(W'*W+delta*eye(K)))/betaDiv(V+eps,W*H+eps,beta));
end
t = toc;
end %EOF
