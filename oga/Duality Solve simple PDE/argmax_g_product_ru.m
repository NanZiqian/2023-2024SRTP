
% Args:
%     un_1:被投影的函数，w,b为g的信息，Cn_1为系数列向量，un_1=(g1,...,gn_1)Cn_1
%     f:simple PDE的右端项
function gn = argmax_g_product_ru(un_1,f)
    syms x;
    % taking ω = ±1 for 1D case
    % ω = (cosθ, sin θ) based on the polar coordinates for 2D case

    % find the best initial samples by evaluating (6.4) at each (bi, wj)
    % then further optimize the best initial sample points using gradient descent or Newton’s method
    % c为神经网络函数sigma(wx+b)中b的取值范围
    bi = linspace(-1,3.02,202)';
    bi(102:202) = bi(102:202)-2.02;
    wi = ones(202,1);
    wi(102:202) = -wi(102:202);

    g = arrayfun(@(func) RELU(func),wi.*x+bi);
    temp=abs(eval( ...
        arrayfun(@(func) GaussInt(func,x,0,1,100),g*un_1) ...
        + arrayfun(@(func) GaussInt(func,x,0,1,100),diff(g)*diff(un_1)) ...
        - arrayfun(@(func) GaussInt(func,x,0,1,100),g*f) ...
        ));
    [~,index] = max(temp);
    gn = g(index);
end
