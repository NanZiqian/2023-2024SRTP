
%% preparation
    clear;
    x=sym("x");
    % -u''+u=f,
    f=(1+pi^2)*cos(pi*x);
    %离散区域数量
    N=100;
    %基的数目，对偶问题中最终基数目为2*BASE_SIZE
    BASE_SIZE=3;

    Ck_1=0;% un_1的系数列向量
    Cprimek_1=0;% phin_1的系数列向量
    w=0;
    b=0;
    k=0;

    %% while
    %while GaussIntegral(abs(u-uk_1),0,1,N) >10^-8 && k < 10^5
    while k < BASE_SIZE
        k=k+1;% 即将产生第2k-1,2k个基

        % g_n=argmax |< g,un_1-u >_H|
        [w_g(k),b_g(k)] = argmax_g_product_ru(Ck_1,w,b,f,1);
        % phi_n=argmax |< g,phin_1-phi >_H|
        [w_h(k),b_h(k)] = argmax_g_product_rphi(Cprimek_1,w,b,1);


        %w前k个为g的信息，后k个为h的信息；缺点：每一轮都得重写
        w=w_g;% g(i)=RELU(x*w_g(i)+b_g(i),1);
        w(1,k+1:2*k)=w_h;% h(i)=RELU(x*w_h(i)+b_h(i),1);
        b=b_g;
        b(1,k+1:2*k)=b_h;

        %w,b包含span(g1,...,gk,h1,...,hk)的信息
        Ck=projection_u(f,w,b);
        Cprimek=projection_phi(w,b);

        %下一轮循环
        Ck_1=Ck;
        Cprimek_1=Cprimek;
    end

    %% 在这之后分别输出un_1,u的图像，进行比较
        % 需要C与w,b才能获得uk=(g1,...,gk,h1,...,hk)Ck
    uk=0;
    for i=1:2*k
        uk=uk+RELU(w(i)*x+b(i),1);
    end
    
    fplot(uk,[0,1],':r');
    hold on
    fplot(cos(pi*x),[0,1],'-b');
