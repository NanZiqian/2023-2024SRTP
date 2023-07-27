
% -u''+u=f,f=(1+pi^2)*cos(pi*x);
%BASE_SIZE need to be even or will be minus one
function [uk] = Duality_approx_simple_PDE(BASE_SIZE)
%% preparation
    x=sym("x");
    % -u''+u=f,f=(1+pi^2)*cos(pi*x);
    f=(1+pi^2)*cos(pi*x);

    %离散区域数量
    % N=100;没用上

    %基的数目，对偶问题中最终基数目为2*BASE_SIZE/2=BASE_SIZE
    BASE_SIZE=round(BASE_SIZE/2);

    uk=0*x;
    phik=0*x;
    gh=x;
    k=0;

    %% while
    %while GaussIntegral(abs(u-uk_1),0,1,N) >10^-8 && k < 10^5
    while k < BASE_SIZE
        k=k+1;% 即将产生第2k-1,2k个基

        % g_n=argmax |< g,un_1-u >_H|
        gh(2*k-1) = argmax_g_product_ru(uk,f);
        % h_n=argmax |< g,phin_1-phi >_H|
        gh(2*k) = argmax_g_product_rphi(phik);

        %w,b包含span(g1,...,gk,h1,...,hk)的信息
        uk=projection_u(f,gh,2*k);
        phik=projection_phi(gh,2*k);
    end

end
%     fplot(uk,[0,1],':r');
%     hold on
%     fplot(cos(pi*x),[0,1],'-b');
