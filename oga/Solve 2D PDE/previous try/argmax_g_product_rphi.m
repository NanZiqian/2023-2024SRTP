
% w,b为g的信息，Cn_1为系数列向量，phin_1=(h1,...,hn)Cn_1
% c为神经网络函数sigma(wx+b)中b的取值范围
function hn = argmax_g_product_rphi(phin_1)
    syms x [2,1];
    
    max=-Inf;
    N=100;
    for wi1=-1:0.4:1
        for wi2=-1:0.4:1
            for bi=-1:0.4:1
                hi=RELU(wi1*x1+wi2*x2+bi,2);
                % 这里没用我们自己写的Gauss积分，因为太慢了
                temp=abs(GaussInt(GaussInt(hi*phin_1,x1,0,1,N),x2,0,1,N) + GaussInt(GaussInt(diff(hi,x1)*diff(phin_1,x1),x1,0,1,N),x2,0,1,N)+GaussInt(GaussInt(diff(hi,x2)*diff(phin_1,x2),x1,0,1,N),x2,0,1,N) - subs(hi,x,[0;0]));
                if temp > max
                    hn=hi;
                    max=temp;
                end
            end
        end
    end
end
