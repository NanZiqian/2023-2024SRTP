
%% solve -u''+u=f,f=(1+pi^2)*cos(pi*x);
function [id,C_g] = OGA_1D_Duality(BASE_SIZE,nd,f)
    % the answer should be cos(pi*x)
    
    %BASE_SIZE = 100; need to be even or will -1
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
    g = max([repmat(qpt,1,nd/2)+b',-repmat(qpt,1,nd/2)+b'],0); % ReLU1
    dg = double(g > 0);% differential of g
    dg(:,nd/2+1:nd) = -dg(:,nd/2+1:nd);
    
    % value of f and solution u at quadrature points
    fqpt = f(qpt);
    uqpt = cos(pi*qpt);
    
    un_1 = zeros(3*N,1);%-u''+u=f
    dun_1 = zeros(3*N,1);
    Phin_1 = zeros(3*N,1);%-Phi''+Phi=l,l(f)=f(0)
    dPhin_1 = zeros(3*N,1);
    
    % matrix and rhs for projection
    A = zeros(BASE_SIZE,BASE_SIZE); 
    rhs_g = zeros(BASE_SIZE,1); 
    rhs_h = zeros(BASE_SIZE,1);
    
    % argmax(i) is <g,u-un_1>H of wx+b(i) of each iteration
    id = zeros(BASE_SIZE,1); argmax_g = zeros(nd,1);argmax_h = zeros(nd,1);
    err = id;
    
    for i = 1:iter % iter = [BASE_SIZE/2]
        for j = 1:nd % <g,u-un_1>H
            argmax_g(j) = norm_L2(g(:,j).*(fqpt-un_1))-norm_L2(dg(:,j).*dun_1);
        end
        for j = 1:nd % <h,Phi-Phin_1>H,RELU
            argmax_h(j) = max(b( mod(j-1,80001)+1 ),0)*1-norm_L2(g(:,j).*Phin_1)-norm_L2(dg(:,j).*dPhin_1);
        end
        [~,id(2*i-1)] = max(abs(argmax_g));% argmax_g = <g,u-un_1>H of all g,argmax(j)-g(:,j)
        [~,id(2*i)] = max(abs(argmax_h));% argmax_h = <h,Phi-Phin_1>H of all h
        for j = 1:2*i
            A(j,2*i) = norm_L2( g(:,id(j)).*g(:,id(2*i)) + dg(:,id(j)).*dg(:,id(2*i)) );
            A(2*i,j) = A(j,2*i);
            if j == 2*i
                break
            end
            A(j,2*i-1) = norm_L2( g(:,id(j)).*g(:,id(2*i-1)) + dg(:,id(j)).*dg(:,id(2*i-1)) );
            A(2*i-1,j) = A(j,2*i-1);
        end
        rhs_g(2*i-1) = norm_L2(g(:,id(2*i-1)).*fqpt);
        rhs_g(2*i) = norm_L2(g(:,id(2*i)).*fqpt);
        C_g = lsqminnorm(A(1:2*i,1:2*i),rhs_g(1:2*i));
        % r = u - un_1
        un_1 = g(:,id(1:2*i))*C_g;
        dun_1 = dg(:,id(1:2*i))*C_g;
        
        rhs_h(2*i-1) = max(b( mod(id(2*i-1)-1,80001)+1 ),0)*1;% int_0^1 g(0)*dx
        rhs_h(2*i) = max(b( mod(id(2*i)-1,80001)+1 ),0)*1;
        C_h = lsqminnorm(A(1:2*i,1:2*i),rhs_h(1:2*i));
        Phin_1 = g(:,id(1:2*i))*C_h;
        dPhin_1 = dg(:,id(1:2*i))*C_h;
        
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