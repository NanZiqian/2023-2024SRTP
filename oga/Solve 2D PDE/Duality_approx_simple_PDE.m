
%% preparation
    clear;
    syms x [2,1];
    % -Δu+u=f
    f=(1+8*pi^2)*cos(2*pi*x(1))*cos(2*pi*x(2));

    %离散区域数量
    %N=100;

    %基的数目，对偶问题中最终基数目为2*BASE_SIZE
    BASE_SIZE=4;

    uk=0;
    phik=0;
    %gh=sym("g",[1,2*BASE_SIZE]);
    k=0;

    %% while
    %while GaussIntegral(abs(u-uk_1),0,1,N) >10^-8 && k < 10^5
    while k < BASE_SIZE
        k=k+1;% 即将产生第2k-1,2k个基

        % g_n=argmax |< g,un_1-u >_H|
        gh(2*k-1) = argmax_g_product_ru(uk,f);
        % phi_n=argmax |< g,phin_1-phi >_H|
        gh(2*k) = argmax_g_product_rphi(phik);

        %w,b包含span(g1,...,gk,h1,...,hk)的信息
        Ck=projection_u(f,gh,2*k);
        Cprimek=projection_phi(gh,2*k);

        % 需要C与gh才能获得uk=(g1,...,gk,h1,...,hk)Ck
        uk=gh*Ck;
        phik=gh*Cprimek;
    end
    
    
    
    %% 在这之后分别输出un_1,u的图像，进行比较
    [a,b]=meshgrid(0:0.1:1,0:0.1:1);
    u=cos(2*pi*a).*cos(2*pi*b);
    uuk=arrayfun(@(p,q) subs(uk,x,[p;q]),a,b);
    surf(a,b,u);
    hold on;
    surf(a,b,uuk);