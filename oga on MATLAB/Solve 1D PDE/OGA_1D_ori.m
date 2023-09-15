
%% solve -u''+u=f,f=(1+pi^2)*cos(pi*x);
% the answer should be cos(pi*x)
% BASE_SIZE = 100
% nd = 160002 number of dictionary
% f = @(z) (1+pi^2)*cos(pi*z);
function [id,C] = OGA_1D_ori(BASE_SIZE,nd,f)
    
    % BASE_SIZE = 100
    iter = BASE_SIZE;
    % f = @(z) (1+pi^2)*cos(pi*z); 
    N = 500; h = 1/N;% N = 500;Gauss integral
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
    % hd = 5e-05
    hd = 8/(nd-2); b = (-2.0:hd:2.0)'; 
    
    %g = [repmat(qpt,1,nd)+b',-repmat(qpt,1,nd)+b']>0; % ReLU0
    %g(i,j)=1*qpt(i)+b(j),i<80001
    %g(i,j+80001)=-1*qpt(i)+b(j-80001)
    g = max([repmat(qpt,1,nd/2)+b',-repmat(qpt,1,nd/2)+b'],0); % ReLU1
    dg = double(g > 0);% differential of g
    dg(:,80002:160002) = -dg(:,80002:160002);
    
    % value of f at quadrature points
    fqpt = f(qpt);
    uqpt = cos(pi*qpt);
    un_1 = zeros(1500,1);
    dun_1 = zeros(1500,1);
    
    % matrix and rhs for projection
    A = zeros(iter,iter); rhs = zeros(iter,1); 
    
    % argmax(i) is <g,u-un_1>H of wx+b(i) of each iteration
    id = zeros(iter,1); argmax = zeros(nd,1);
    
    err = id;
    for i = 1:iter
        for j = 1:nd % number of dictionary
            argmax(j) = norm_L2(g(:,j).*fqpt) - norm_L2(g(:,j).*un_1)-norm_L2(dg(:,j).*dun_1);
        end
        [~,id(i)] = max(abs(argmax));% optimal b of i^th iteration
        for j = 1:i
            A(j,i) = norm_L2(g(:,id(j)).*g(:,id(i))) + norm_L2(dg(:,id(j)).*dg(:,id(i)));
            A(i,j) = A(j,i);
        end
        rhs(i) = norm_L2(g(:,id(i)).*fqpt);
        C = lsqminnorm(A(1:i,1:i),rhs(1:i));
        un_1 = g(:,id(1:i))*C;% r = u - un_1
        dun_1 = dg(:,id(1:i))*C;

        r = uqpt - un_1;
        err(i) = sqrt(norm_L2(r.^2));
        fprintf("Step %d, error_L2 is %f\n",i,err(i));
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
% F(1:end/3) = function F
% F(i)=
function z = norm_L2(F)
    z = 5/18*sum( F(1:end/3) )+4/9*sum( F(end/3+1:end/3*2) )+5/18*sum( F(end/3*2+1:end) );
    z = z/500;% 500 == N is Gauss quadrature discretion number
end