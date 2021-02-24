function x = SimplexProjW(y)

% Given y,  computes its projection x* onto the simplex 
% 
%       Delta = { x | x >= 0 and sum(x) <= 1 }, 
% 
% that is, x* = argmin_x ||x-y||_2  such that  x in Delta. 
% 
%  
% See Appendix A.1 in N. Gillis, Successive Nonnegative Projection 
% Algorithm for Robust Nonnegative Blind Source Separation, arXiv, 2013. 
% 
%
% x = SimplexProj(y)
%
% ****** Input ******
% y    : input vector.
%
% ****** Output ******
% x    : projection of y onto Delta.

x = max(y,0); 
%K = find(sum(x) > 1); 
%x(:,K) = blockSimplexProj(y(:,K));
x = blockSimplexProj(y);

end


function x = blockSimplexProj(y)

% Same as function SimplexProj except that sum(max(Y,0)) > 1. 
[r,m] = size(y); 
ys = sort(-y);  ys = -ys;
indi2 = 1:m; lambda = zeros(1,m); 
S(1,indi2) = 0; 
for i = 2 : r
    if i == 2
        S(i,:) = (ys(1:i-1,:)-repmat(ys(i,:),i-1,1)); 
    else
        S(i,:) = sum(ys(1:i-1,:)-repmat(ys(i,:),i-1,1)); 
    end
    indi1 = find(S(i,indi2) >= 1); 
    indi1 = indi2(indi1);
    indi2 = find(S(i,:) < 1);
    if ~isempty(indi1)
        if i == 1
            lambda(indi1) = -ys(1,indi1)+1;
        else
            lambda(indi1) = (1-S(i-1,indi1))/(i-1) - ys(i-1,indi1);
        end
    end
    if i == r
        lambda(indi2) = (1-S(r,indi2))/r - ys(r,indi2);
    end
end
x = max( y + repmat(lambda,r,1), 0); 
end