
%% solve -u''+u=f,f=(1+pi^2)*cos(pi*x),u'(-1)=u'(1)=0;
function [w_Newton,b_Newton,C_g] = OGA_Newton_1D_Duality(BASE_SIZE,nd,f)
    % the answer should be cos(pi*x)
    
    %BASE_SIZE need to be even or will be -1
    iter = floor(BASE_SIZE/2);
    % f = @(x) (1+pi^2)*cos(pi*x); 
    N = 500; h = 1/N;
    node = (0:h:1)';% column
    
    % 3-point Gauss quadrature
    c = [1/2-sqrt(15)/10 1/2 1/2+sqrt(15)/10];
    
    % quadrature points devide each 1/N into three. 
    % qpt(1:500,1):[x1+c(1)*h;x2+c(1)*h;...]
    % qpt(500:1000,1):[x1+c(2)*h;x2+c(2)*h;...]
    % ...
    qpt = [node(1:end-1)+c(1)*h;node(1:end-1)+c(2)*h;node(1:end-1)+c(3)*h];
    
    % discretization of the dictionary
    % nd = 160002;number of dictionary
    hd = 8/(nd-2); b = (-2.0:hd:2.0)'; 
    
    %g = [repmat(qpt,1,nd)+b',-repmat(qpt,1,nd)+b']>0; % ReLU0
    %g(i,j)=1*qpt(i)+b(j),i<80001
    %g(i,j+80001)=-1*qpt(i)+b(j-80001)
    g_fixed = max([repmat(qpt,1,nd/2)+b',-repmat(qpt,1,nd/2)+b'],0); % ReLU1
    dg_fixed = double(g_fixed > 0);% differential of g
    dg_fixed(:,nd/2+1:nd) = -dg_fixed(:,nd/2+1:nd);
    g = @(w,b) max(w*qpt+b,0);
    dg_x = @(w,b) w*double(w*qpt+b>0);
    dg_b = @(w,b) double(w*qpt+b>0);
    
    % value of f and solution u at quadrature points
    fqpt = f(qpt);
    uqpt = cos(pi*qpt);
    
    un_1 = zeros(3*N,1);%-u''+u=f
    dun_1 = zeros(3*N,1);
    Phin_1 = zeros(3*N,1);%-Phi''+Phi=l,l(f)=f(0)
    dPhin_1 = zeros(3*N,1);
    g_base = zeros(3*N,BASE_SIZE);
    dg_base = g_base;
    
    % matrix and rhs for projection
    A = zeros(BASE_SIZE,BASE_SIZE); 
    rhs_g = zeros(BASE_SIZE,1); 
    rhs_h = zeros(BASE_SIZE,1);
    
    % argmax(i) is <g,u-un_1>H of wx+b(i) of each iteration
    id = zeros(BASE_SIZE,1); 
    w_Newton = id;
    b_Newton = id;
    err = id;
    argmax_g = zeros(nd,1);
    argmax_h = zeros(nd,1);
    
    for i = 1:iter % iter = [BASE_SIZE/2]
        for j = 1:nd % <g,u-un_1>H
            argmax_g(j) = norm_L2(g_fixed(:,j).*(fqpt-un_1))-norm_L2(dg_fixed(:,j).*dun_1);
        end
        for j = 1:nd % <h,Phi-Phin_1>H,RELU
            argmax_h(j) = max(b( mod(j-1,80001)+1 ),0)*1-norm_L2(g_fixed(:,j).*Phin_1)-norm_L2(dg_fixed(:,j).*dPhin_1);
        end
        [~,id(2*i-1)] = max(abs(argmax_g));% argmax_g = <g,u-un_1>H of all g,argmax(j)-g(:,j)
        [~,id(2*i)] = max(abs(argmax_h));% argmax_h = <h,Phi-Phin_1>H of all h
        %% use Gredient descent to find argmax
        % argmax_g
        if id(2*i-1)>nd/2
            w_Newton(2*i-1) = -1;
            b_Newton(2*i-1) = b( mod(id(2*i-1)-1,nd/2)+1 );
        else
            w_Newton(2*i-1) = 1;
            b_Newton(2*i-1) = b( id(2*i-1) );
        end
        F = @(w,b) -1/2*( norm_L2(g(w,b) .* (fqpt-un_1)) - norm_L2(dg_x(w,b).*dun_1) )^2;
        dF_b = @(w,b) -(norm_L2(g(w,b) .* (fqpt-un_1)) - norm_L2(dg_x(w,b).*dun_1))*( norm_L2(dg_b(w,b).*(fqpt-un_1)) );
        
        bk = b_Newton(2*i-1);
        while norm(dF_b(1,bk),2) >= 1e-5
            % need to set the range of b, this time [-2,2]
            [sign,alpha] = armijo(w_Newton(2*i-1),bk,F,dF_b,-2,2);
            temp = bk-alpha*dF_b(w_Newton(2*i-1),bk);
            if sign && abs(temp)<=2
                bk = temp;
            else
                break;
            end
        end
        b_Newton(2*i-1) = bk;
        g_base(:,2*i-1) = g(w_Newton(2*i-1),bk);
        dg_base(:,2*i-1) = dg_x(w_Newton(2*i-1),bk);

        % argmax_h
        if id(2*i)>nd/2
            w_Newton(2*i) = -1;
            b_Newton(2*i) = b( mod(id(2*i)-1,nd/2)+1 );
        else
            w_Newton(2*i) = 1;
            b_Newton(2*i) = b( id(2*i) );
        end
        F = @(w,b) -1/2*( max(b,0)-norm_L2(g(w,b) .* un_1) - norm_L2(dg_x(w,b).*dun_1) )^2;
        dF_b = @(w,b) -(max(b,0)-norm_L2(g(w,b) .* un_1) - norm_L2(dg_x(w,b).*dun_1)) * (double(b>0)-norm_L2(dg_b(w,b).*un_1));
        
        bk = b_Newton(2*i);
        while norm(dF_b(1,bk),2) >= 1e-5
            % need to set the range of b, this time [-2,2]
            [sign,alpha] = armijo(w_Newton(2*i),bk,F,dF_b,-2,2);
            temp = bk-alpha*dF_b(w_Newton(2*i),bk);
            if sign && abs(temp)<=2
                bk = temp;
            else
                break;
            end
        end
        b_Newton(2*i) = bk;
        g_base(:,2*i) = g(w_Newton(2*i),bk);
        dg_base(:,2*i) = dg_x(w_Newton(2*i),bk);

        %% u_n = Pn(u)
        for j = 1:2*i
            A(j,2*i) = norm_L2( g_base(j).*g_base(2*i) + dg_base(j).*dg_base(2*i) );
            A(2*i,j) = A(j,2*i);
            if j == 2*i
                break
            end
            A(j,2*i-1) = norm_L2( g_base(j).*dg_base(2*i-1) + dg_base(j).*dg_base(2*i-1) );
            A(2*i-1,j) = A(j,2*i-1);
        end
        rhs_g(2*i-1) = norm_L2(dg_base(2*i-1).*fqpt);
        rhs_g(2*i) = norm_L2(g_base(2*i).*fqpt);
        C_g = lsqminnorm(A(1:2*i,1:2*i),rhs_g(1:2*i));
        % r = u - un_1
        un_1 = g_base(1:2*i)*C_g;
        dun_1 = dg_base(1:2*i)*C_g;
        
        rhs_h(2*i-1) = max(b_Newton(2*i-1),0);% int_0^1 g(0)*dx
        rhs_h(2*i) = max(b_Newton(2*i),0);
        C_h = lsqminnorm(A(1:2*i,1:2*i),rhs_h(1:2*i));
        Phin_1 = g_base(1:2*i)*C_h;
        dPhin_1 = dg_base(1:2*i)*C_h;
        
        r = uqpt - un_1;
        err(i) = sqrt(norm_L2(r.^2));
        fprintf("Step %d, error_L2 is %f\n",2*i,err(i));
    end
end

%% draw
% iter = 100;
% subplot(1,2,1);
% syms x;
% fplot(cos(pi*x),[0,1],'-b');
% hold on
% plot(qpt,un_1,':r');
% 
% subplot(1,2,2);
% iter = (1:iter)';
% 
% plot(log(iter),log10(err),'.');
% hold on 
% %plot(log(iter),-1*log10(iter)-0.6,'-.'); % k=0 
% plot(log(iter),-2*log10(iter)-1.5,'-.'); % k=1
% 
% st = 10;
% temp = polyfit(log(iter(st:end)),log(err(st:end)),1);
% fprintf('The convergence rate is %.2e \n', -temp(1));

%% --------------------------------------------------------------------------
% F(1:end/3) = F(x_i+c(1)*h)
% not really norm_L2, actually is Gauss integral, need to product two
% functions before-hand
function z = norm_L2(F)
z = 5/18*sum( F(1:end/3) )+4/9*sum( F(end/3+1:end/3*2) )+5/18*sum( F(end/3*2+1:end) );
z = z/500;% 500 == N is Gauss quadrature discretion number
end

function [sign,alpha] = armijo(w,xk,f,gradf,a,b)
    % xk must be in [a,b]
    % xk as a parameter is x0
    m=0;
    beta = 0.5;
    sigma = 0.2;
    gk = gradf(w,xk);
    dk = -gk;

    sign = 1;
    while 1
        alpha = beta^m;
        if xk == b && dk>0 || xk == a && dk<0
            break;
        end
        temp = xk+alpha*dk;
        if temp>b || temp<a
            m=m+1;
        else
            break;
        end
    end
    while 1
        alpha = beta^m;
        if f(w,xk+alpha*dk) <= f(w,xk) + sigma*alpha*gk'*dk
            break;
        end
        m = m+1;
        if m > 53
            % unable to find alpha
            sign = 0;
            break;
        end
    end
end