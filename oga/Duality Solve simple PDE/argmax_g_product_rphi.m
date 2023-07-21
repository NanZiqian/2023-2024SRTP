
% w,b为g的信息，Cn_1为系数列向量，phin_1=(h1,...,hn)Cn_1
% c为神经网络函数sigma(wx+b)中b的取值范围
function [wk,bk] = argmax_g_product_rphi(Cn_1,w,b,c)
    syms x;
    hi=arrayfun(@(x) RELU(x,1),w.*x+b);
    phin_1=hi*Cn_1;
    max=-Inf;

    for wi=-1:2*0.1:1
        for bi=-c:2*c/10:c
            g=RELU(wi*x+bi,1);
            % 这里没用我们自己写的Gauss积分，因为太慢了
            temp=abs( int(g*phin_1, x ,0,1) + int(diff(g)*diff(phin_1), x,0,1 ) - eval(subs(g,x,0)) );
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
