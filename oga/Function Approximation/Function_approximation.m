
% u=(1+pi^2)*cos(pi*x)
function [uk]=Function_approximation(BASE_SIZE)
%% preparation
    x=sym("x");
    u=(1+pi^2)*cos(pi*x);%近似的函数
    %N=100; 离散区域数量积分用
    uk_1=0;
    Ck_1=0;
    w=0;
    b=0;
    k=0;

    %% while
    %while GaussIntegral(abs(u-uk_1),0,1,N) >10^-8 && k < 10^5
    while k < BASE_SIZE
        k=k+1;% 即将产生第k个基

        % 需要C与w,b才能获得uk_1=(g1,...,gk_1)C
        [w(k),b(k)] = argmax_g_product_r(Ck_1,w,b,u,1);
        
        %w,b包含span(g1,...,gk)的信息
        Ck=projection(u,w,b);
        
        gi(k)=RELU(w(k)*x+b(k),2);
        %下一轮循环
        uk_1=uk;
        Ck_1=Ck;
    end
    uk=gi*Ck;
end
    %% 在这之后分别输出un_1,u的图像，进行比较
%     fplot(uk,[0,1],':r');
%     hold on
%     fplot(u,[0,1],'-b');