%% approximate the function f on [0,1] with iter OGA iterations
iter = 100;
f = @(z) sin(z); 
N = 500; h = 1/N;
node = (0:h:1)';

c = [1/2-sqrt(15)/10 1/2 1/2+sqrt(15)/10];% 3-point Gauss quadrature
% quadrature points devide each 1/N into three. the first is qpt(1:500,1)
qpt = [node(1:end-1)+c(1)*h;node(1:end-1)+c(2)*h;node(1:end-1)+c(3)*h];

hd = 5e-5; b = (-2.0:hd:2.0)'; % discretization of the dictionary
nd = length(b)*2;

%g = [repmat(qpt,1,nd)+b',-repmat(qpt,1,nd)+b']>0; % ReLU0
%g(i,j)=1*qpt(i)+b(j),i<80001
%g(i,j+80001)=-1*qpt(i)+b(j-80001)
g = max([repmat(qpt,1,nd/2)+b',-repmat(qpt,1,nd/2)+b'],0); % ReLU1

% value of f at quadrature points
fqpt = f(qpt); r = fqpt;

% inner products of selected dictionary elements.
A = zeros(iter,iter); rhs = zeros(iter,1); % matrix and rhs for projection

% argmax(i) is <g,u-un_1>H of +-x+b(i) of each iteration
id = zeros(iter,1); argmax = zeros(nd,1);

err = id;
for i = 1:iter
    for j = 1:nd % number of b
        argmax(j) = pnorm(g(:,j).*r);% r = f(qpt)
    end
    [~,id(i)] = max(abs(argmax));% optimal b of i^th iteration
    for j = 1:i
        A(j,i) = pnorm(g(:,id(j)).*g(:,id(i)));
        A(i,j) = A(j,i);
    end
    rhs(i) = pnorm(g(:,id(i)).*fqpt);
    C = lsqminnorm(A(1:i,1:i),rhs(1:i));
    r = fqpt - g(:,id(1:i))*C;% r = u - un_1

    err(i) = sqrt(pnorm(r.^2));

    fprintf("Step %d, error is %f\n",i,err(i));
end

%% draw
subplot(1,2,1);
syms x;
un_1 = fqpt - r;
fplot(sin(x),[0,1],'-b');
hold on
plot(qpt,un_1,':r');

subplot(1,2,2);
iter = (1:iter)';

plot(log(iter),log10(err),'.');
hold on 
%plot(log(iter),-1*log10(iter)-0.6,'-.'); % k=0 
plot(log(iter),-2*log10(iter)-1.5,'-.'); % k=1

st = 10;
temp = polyfit(log(iter(st:end)),log(err(st:end)),1);
fprintf('The convergence rate is %.2e \n', -temp(1));
%% --------------------------------------------------------------------------
function z = pnorm(F)
z = 5/18*sum( F(1:end/3) )+4/9*sum( F(end/3+1:end/3*2) )+5/18*sum( F(end/3*2+1:end) );
z = z/500;
end





