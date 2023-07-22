
% w,b为g的信息，Cn_1为系数列向量，phin_1=(h1,...,hn)Cn_1
% c为神经网络函数sigma(wx+b)中b的取值范围
function hn = argmax_g_product_rphi(phin_1)
    x=sym("x",[2,1]);
    max=-Inf;

    for wi=[-1:0.2:1,-1:0.2:1]
        for bi=-2:0.4:2
            hi=RELU(wi*x+bi,1);
            % 这里没用我们自己写的Gauss积分，因为太慢了
            temp=abs( int(hi*phin_1, x ,0,1) + int(diff(hi)*diff(phin_1), x,0,1 ) - eval(subs(hi,x,0)) );
            if temp > max
                hn=hi;
                max=temp;
            end
        end
    end
end
