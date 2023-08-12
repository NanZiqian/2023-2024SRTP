
% -u''+u=f,f=(1+pi^2)*cos(pi*x);
function [uk] = Approx_simple_PDE(BASE_SIZE)
%% preparation
    x=sym("x");
    % -u''+u=f
    f=(1+pi^2)*cos(pi*x);

    %N=100;离散区域数量
    uk=0*x;
    g=x;
    k=0;

    %% while
    while k < BASE_SIZE
        k=k+1;% 即将产生第k个基

        % g_n=argmax |< g,un_1-u >_H|
        g(k) = argmax_g_product_r(uk,f);
        
        %u在(g1,...,gk)的投影即为uk
        uk=projection(f,g,k);
    end

end

    %% 在这之后分别输出un_1,u的图像，进行比较
%     fplot(uk,[0,1],':r');
%     hold on
%     fplot(cos(pi*x),[0,1],'-b');
