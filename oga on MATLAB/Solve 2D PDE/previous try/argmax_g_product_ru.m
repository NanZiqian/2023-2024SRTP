
%   w,b为g的信息，Cn_1为系数列向量，un_1=(g1,...,gn_1)Cn_1
% c为神经网络函数sigma(wx+b)中b的取值范围
function gn = argmax_g_product_ru(un_1,f)
    syms x [2,1];
    
    max=-Inf;
    N=100;
    for wi1=-1:0.4:1
        for wi2=-1:0.4:1
            for bi=-1:0.4:1
                gi=RELU(wi1*x1+wi2*x2+bi,2);
                % 这里没用我们自己写的Gauss积分，因为太慢了
                temp=abs(GaussInt(GaussInt(gi*un_1,x1,0,1,N),x2,0,1,N)+GaussInt(GaussInt(diff(gi,x1)*diff(un_1,x1),x1,0,1,N),x2,0,1,N)+GaussInt(GaussInt(diff(gi,x2)*diff(un_1,x2),x1,0,1,N),x2,0,1,N)-GaussInt(GaussInt(gi*f,x1,0,1,N),x2,0,1,N));
                if temp > max
                    gn=gi;
                    max=temp;
                end
            end
        end
    end
end
