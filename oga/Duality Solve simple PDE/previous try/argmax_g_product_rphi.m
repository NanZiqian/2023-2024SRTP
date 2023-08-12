
% w,b为g的信息，Cn_1为系数列向量，phin_1=(h1,...,hn)Cn_1
% c为神经网络函数sigma(wx+b)中b的取值范围
function hn = argmax_g_product_rphi(phin_1)
    syms x;
    
    bi = linspace(-1,3.02,202)';
    bi(102:202) = bi(102:202)-2.02;
    wi = ones(202,1);
    wi(102:202) = -wi(102:202);

    h = arrayfun(@(func) RELU(func),wi.*x+bi);
    temp=abs(eval( ...
        arrayfun(@(func) GaussInt(func,x,0,1,100),h*phin_1) + ...
        arrayfun(@(func) GaussInt(func,x,0,1,100),diff(h)*diff(phin_1)) ...
        - subs(h,x,0)...
        ));
    [~,index] = max(temp);
    hn = h(index);
end

