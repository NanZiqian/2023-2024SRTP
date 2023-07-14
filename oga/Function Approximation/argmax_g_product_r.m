
%   w,b为g的信息，Cn_1为系数列向量C;真实的un_1=(g1,...,gn)C
% c为神经网络函数sigma(wx+b)中b的取值范围
function [wk,bk] = argmax_g_product_r(Cn_1,w,b,u,c)
    syms x;
    u=subs(u,x);
    [~,n]=size(w);
    for i=1:n
        gi(i)=RELU(w(i)*x+b(i),1);
    end
    un_1=gi*Cn_1;
    max=-Inf;

    for wi=-1:2*0.1:1
        for bi=-c:2*c/10:c
            g=RELU(wi*x+bi,1);
            % 这里没用我们自己写的Gauss积分，因为太慢了
            temp=abs(int(g*(un_1-u),x,0,1));
            if temp > max
                w_star=wi;
                b_star=bi;
                max=temp;
            end
        end
    end
    wk=w_star;
    bk=b_star;
end

