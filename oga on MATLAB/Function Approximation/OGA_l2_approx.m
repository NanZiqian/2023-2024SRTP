%% approximate the function f on [0,1] with iter OGA iterations
iter = 100;
f = @(z) sin(z); 
N = 500; h = 1/N;
node = (0:h:1)';

c = [1/2-sqrt(15)/10 1/2 1/2+sqrt(15)/10];% 3-point Gauss quadrature
qpt = [node(1:end-1)+c(1)*h;node(1:end-1)+c(2)*h;node(1:end-1)+c(3)*h];

hd = 5e-5; b = (-2.0:hd:2.0)'; % discretization of the dictionary
nd = length(b);

%g = [repmat(qpt,1,nd)+b',-repmat(qpt,1,nd)+b']>0; % ReLU0
g = max([repmat(qpt,1,nd)+b',-repmat(qpt,1,nd)+b'],0); % ReLU1
  
fqpt = f(qpt); r = fqpt;
A = zeros(iter,iter); rhs = zeros(iter,1); % matrix and rhs for projection
id = zeros(iter,1); argmax = zeros(nd,1);
err = id;
for i = 1:iter
    for j = 1:nd
        argmax(j) = h*pnorm(g(:,j).*r);
    end
    [~,id(i)] = max(abs(argmax));
    for j = 1:i
        A(j,i) = h*pnorm(g(:,id(j)).*g(:,id(i)));
        A(i,j) = A(j,i);
    end
    rhs(i) = h*pnorm(g(:,id(i)).*fqpt);
    C = lsqminnorm(A(1:i,1:i),rhs(1:i));
    r = fqpt - g(:,id(1:i))*C;
    err(i) = sqrt(h*pnorm(r.^2));
    fprintf("Step %d, error is %f\n",i,err(i));
end

iter = (1:iter)';
plot(log(iter),log10(err),'.');
hold on 
%plot(log(iter),-1*log10(iter)-0.6,'-.'); % k=0 
plot(log(iter),-2*log10(iter)-1.5,'-.'); % k=1
st = 10;
temp = polyfit(log(iter(st:end)),log(err(st:end)),1);
fprintf('The convergence rate is %.2e \n', -temp(1));
%--------------------------------------------------------------------------
function z = pnorm(F)
z = 5/18*sum( F(1:3:end) )+4/9*sum( F(2:3:end) )+5/18*sum( F(3:3:end) );
end




