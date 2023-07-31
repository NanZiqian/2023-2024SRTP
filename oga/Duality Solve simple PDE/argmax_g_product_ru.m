
%   w,b为g的信息，Cn_1为系数列向量，un_1=(g1,...,gn_1)Cn_1
% c为神经网络函数sigma(wx+b)中b的取值范围
function gn = argmax_g_product_ru(un_1,f)
    syms x;
    
    max=-Inf;

    for wi=-1:0.2:1
        for bi=-1:0.2:1
            g=RELU(wi*x+bi,2);
            % 这里没用我们自己写的Gauss积分，因为太慢了
            temp=abs( eval(int(g*un_1, x ,0,1) + int(diff(g)*diff(un_1), x,0,1 ) - int(g*f,x,0,1)));
            if temp > max
                gn=g;
                max=temp;
            end
        end
    end

end
