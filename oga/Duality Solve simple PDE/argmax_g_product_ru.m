
%   w,b为g的信息，Cn_1为系数列向量，un_1=(g1,...,gn_1)Cn_1
% c为神经网络函数sigma(wx+b)中b的取值范围
function [w_star,b_star] = argmax_g_product_ru(Cn_1,w,b,f,c)
    syms x;
    gi=arrayfun(@(x) RELU(x,2),w.*x+b);
    un_1=gi*Cn_1;
    max=-Inf;

    for wi=-1:2*0.1:1
        for bi=-c:2*c/10:c
            g=RELU(wi*x+bi,2);
            % 这里没用我们自己写的Gauss积分，因为太慢了
            temp=abs( int(g*un_1, x ,0,1) + int(diff(g)*diff(un_1), x,0,1 ) - int(g*f,x,0,1));
            if temp > max
                w_star=wi;
                b_star=bi;
                max=temp;
            end
        end
    end
end
