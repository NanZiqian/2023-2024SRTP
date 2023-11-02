
N = 500; h = 1/N;
node = (0:h:1)';% column
% 3-point Gauss quadrature
c = [1/2-sqrt(15)/10 1/2 1/2+sqrt(15)/10];
qpt = [node(1:end-1)+c(1)*h;node(1:end-1)+c(2)*h;node(1:end-1)+c(3)*h];
f = @(z) (1+pi^2)*cos(pi*z);
g = @(w,b) max(w*qpt+b,0);
dg_x = @(w,b) w*double(w*qpt+b>0);
dg_b = @(w,b) double(w*qpt+b>0);
fqpt = f(qpt);

un_1 = zeros(3*N,1);%-u''+u=f
dun_1 = zeros(3*N,1);

F = @(w,b) -1/2*( norm_L2(g(w,b) .* (fqpt-un_1)) - norm_L2(dg_x(w,b).*dun_1) )^2;
%dF_b = @(w,b) -(norm_L2(g(w,b) .* (fqpt-un_1)) - norm_L2(dg_x(w,b).*dun_1))*( norm_L2(dg_b(w,b).*(fqpt-un_1)) );
dF_b = @(w,b) -(max(b,0)-norm_L2(g(w,b) .* un_1) - norm_L2(dg_x(w,b).*dun_1)) * (double(b>0)-norm_L2(dg_b(w,b).*un_1));
dF_b(1,0.05)
% bk = -1:0.05:1;
% for i = 1:2/0.05+1
%     temp(i) = double(dF_b(1,bk(i)));
% end
% bk = 0.05 ;
% while norm(dF_b(1,bk),2) >= 1e-5
%     alpha = armijo(bk,F,dF_b);
%     bk = bk-alpha*dF_b(bk);
% end


function z = norm_L2(F)
    z = 5/18*sum( F(1:end/3) )+4/9*sum( F(end/3+1:end/3*2) )+5/18*sum( F(end/3*2+1:end) );
    z = z/500;% 500 == N is Gauss quadrature discretion number
end

function [alpha] = armijo(xk,f,gradf)
    m=0;
    beta = 0.5;
    sigma = 0.2;
    gk = gradf(xk);
    dk = -gk;
    alpha = 1;
    while 1
        alpha = beta^m;
        if f(xk+alpha*dk) <= f(xk) + sigma*alpha*gk'*dk
            break;
        end
        m = m+1;
    end
end