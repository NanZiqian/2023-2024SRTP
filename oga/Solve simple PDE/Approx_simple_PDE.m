
%% preparation
    clear;
    x=sym("x");
    % -u''+u=f
    f=(1+pi^2)*cos(pi*x);

    N=100;%离散区域数量
    Ck=0;
    w=0;
    b=0;
    k=0;

    %% while
    %while GaussIntegral(abs(u-uk_1),0,1,N) >10^-8 && k < 10^5
    while k < 5
        k=k+1;% 即将产生第k个基

        % g_n=argmax |< g,un_1-u >_H|
        [w(k),b(k)] = argmax_g_product_r(Ck,w,b,f,1);
        
        %w,b包含span(g1,...,gk)的信息
        Ck=projection(f,w,b);
        
        gi(k)=RELU(w(k)*x+b(k),1);
    end

    % 需要C与w,b才能获得uk_1=(g1,...,gk_1)C
    uk=gi*Ck;

    %% 在这之后分别输出un_1,u的图像，进行比较
    fplot(uk,[0,1],':r');
    hold on
    fplot(cos(pi*x),[0,1],'-b');
