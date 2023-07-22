
%   w,b为g的信息，Cn_1为系数列向量，un_1=(g1,...,gn_1)Cn_1
% c为神经网络函数sigma(wx+b)中b的取值范围
function gn = argmax_g_product_ru(un_1,f)
    x=sym("x",[2,1]);
    max=-Inf;

    for wi=[-1:0.2:1,-1:0.2:1]
        for bi=-2:0.4:2
            gi=RELU(wi*x+bi,1);
            % 这里没用我们自己写的Gauss积分，因为太慢了
            temp=abs( int(gi*un_1, x ,0,1) + int(diff(gi)*diff(un_1), x,0,1 ) - int(gi*f,x,0,1));
            if temp > max
                gn=gi;
                max=temp;
            end
        end
    end
end
