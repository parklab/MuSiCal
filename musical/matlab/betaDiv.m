function bD = betaDiv(V,Vh,beta)
    if beta == 0
        bD = sum((V(:)./Vh(:))-log(V(:)./Vh(:)) - 1);
    elseif beta==1
        bD = sum(V(:).*(log(V(:))-log(Vh(:))) + Vh(:) - V(:));
    else
        bD = sum(max(1/(beta*(beta-1))*(V(:).^beta + (beta-1)*Vh(:).^beta - beta*V(:).*Vh(:).^(beta-1)),0));
    end
end %EOF