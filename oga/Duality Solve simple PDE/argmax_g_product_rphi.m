
% w,b为g的信息，Cn_1为系数列向量，phin_1=(h1,...,hn)Cn_1
% c为神经网络函数sigma(wx+b)中b的取值范围
function hn = argmax_g_product_rphi(phin_1)
    syms x;
    
    max=-Inf;

    for wi=-1:0.2:1
        for bi=-1:0.2:1
            h=RELU(wi*x+bi,2);
            % 这里没用我们自己写的Gauss积分，因为太慢了
            temp=abs( eval(int(h*phin_1, x ,0,1) + int(diff(h)*diff(phin_1), x,0,1 ) - subs(h,x,0)) );
            if temp > max
                hn=h;
                max=temp;
            end
        end
    end

end
