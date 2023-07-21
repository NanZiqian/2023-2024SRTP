
%% preparation
    clear;
    x=sym("x");
    % -u''+u=f,
    f=(1+pi^2)*cos(pi*x);
    %离散区域数量
    N=100;
    %基的数目，对偶问题中最终基数目为2*BASE_SIZE
    BASE_SIZE=4;

    Ck=0;% un_1的系数列向量
    Cprimek=0;% phin_1的系数列向量
    w=0;
    b=0;
    w_g=zeros(1,BASE_SIZE);
    w_h=zeros(1,BASE_SIZE);
    b_g=zeros(1,BASE_SIZE);
    b_h=zeros(1,BASE_SIZE);
    k=0;

    %% while
    %while GaussIntegral(abs(u-uk_1),0,1,N) >10^-8 && k < 10^5
    while k < BASE_SIZE
        k=k+1;% 即将产生第2k-1,2k个基

        % g_n=argmax |< g,un_1-u >_H|
        [w_g(k),b_g(k)] = argmax_g_product_ru(Ck,w,b,f,1);
        % phi_n=argmax |< g,phin_1-phi >_H|
        [w_h(k),b_h(k)] = argmax_g_product_rphi(Cprimek,w,b,1);


        %w前k个为g的信息，后k个为h的信息；缺点：每一轮都得重写
        w=[w_g(1,1:k),w_h(1,1:k)];% g(i)=RELU(x*w_g(i)+b_g(i),1);h(i)=RELU(x*w_h(i)+b_h(i),1);
        b=[b_g(1,1:k),b_h(1,1:k)];

        %w,b包含span(g1,...,gk,h1,...,hk)的信息
        Ck=projection_u(f,w,b);
        Cprimek=projection_phi(w,b);

    end

    %% 在这之后分别输出un_1,u的图像，进行比较
    % 需要C与w,b才能获得uk=(g1,...,gk,h1,...,hk)Ck
    uk=arrayfun(@(x) RELU(x,1),w.*x+b)*Ck;
    
    fplot(uk,[0,1],':r');
    hold on
    fplot(cos(pi*x),[0,1],'-b');
