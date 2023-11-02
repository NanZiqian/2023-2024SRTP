
F = @(w,b) -1/2*( norm_L2(g(w,b) .* (fqpt-un_1)) - norm_L2(dg_x(w,b).*dun_1) )^2;
dF_b = @(w,b) -(norm_L2(g(w,b) .* (fqpt-un_1)) - norm_L2(dg_x(w,b).*dun_1))*( norm_L2(dg_b(w,b).*(fqpt-un_1)) );

norm_L2(dg_b(1,0.05).*fqpt)
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