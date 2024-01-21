%% solve 1d equation -Δu+u=f on [-1 1] with OGA iterations
clear;
%source function -u''+u=f,f=(1+pi^2)*cos(pi*x);
f=@(x) (1+pi^2)*cos(pi*x); 
u=@(x) cos(pi*x);
iter=300;%迭代次数

N=500;%复化积分区间划分数量
h=1/N;%区间长度
node=(0:h:1);
T=0.5*[1-sqrt((15+2*sqrt(30))/35) 1-sqrt((15-2*sqrt(30))/35) 1+sqrt((15-2*sqrt(30))/35) 1+sqrt((15+2*sqrt(30))/35)];%高斯积分的节点
L=4*N;%总节点数量
qpt=zeros(L,1);%quadrate point 高斯积分节点向量
qpt(1:4:end-3)=node(1:end-1)+T(1)*h;
qpt(2:4:end-2)=node(1:end-1)+T(2)*h;
qpt(3:4:end-1)=node(1:end-1)+T(3)*h;
qpt(4:4:end)=node(1:end-1)+T(4)*h;

hb=1e-2;%b的步长
nd=int64(2*(4/hb+1));%字典中函数数量
g=zeros(L,nd);%离散化的字典 g=ReLU^2(wx+b)
dg=zeros(L,nd);%g对x求导的导函数

j=1;
for w=[-1,1]
    for b=(-2:hb:2)
        wqb=w*qpt+b;
        g(:,j)=max(0,wqb).^2;%g=ReLU^2(wx+b)
        dg(:,j)=w*max(0,2*wqb);
        j=j+1;
    end
end

fqpt=f(qpt);
uq=u(qpt);
uk=zeros(L,1);%数值解
du=zeros(L,1);%uk对x求导的导函数
id = zeros(iter,1);%标记字典中组成uk的基的索引
err=zeros(iter,1);
argmax=zeros(nd,1);%存储|<g,u-uk>|的值
A = zeros(iter,iter); rhs = zeros(iter,1);%求解投影的线性方程组系数矩阵与常数项矩阵

for i=1:iter 
    for j=1:nd
        argmax(j)=int1d(g(:,j).*(uk-fqpt)+dg(:,j).*du)*h;
    end
    [~,id(i)] = max(abs(argmax));
    
    for j = 1:i
        A(j,i) = int1d( g(:,id(j)).*g(:,id(i)) + dg(:,id(j)).*dg(:,id(i)) )*h;
        A(i,j) = A(j,i);
    end
    rhs(i) = int1d(g(:,id(i)).*fqpt)*h;
    C = lsqminnorm(A(1:i,1:i),rhs(1:i));
    %C=A(1:2*i,1:2*i)\rhs(1:2*i);
    uk=zeros(L,1);%数值解
    du=zeros(L,1);%uk对x求导的导函数
    for j = 1:i
        uk=uk+g(:,id(j))*C(j);
        du=du+dg(:,id(j))*C(j);
    end
    r=uq-uk;
    err(i) = sqrt(int1d(r.^2)*h);
    fprintf("Step %d, error_L2 is %f\n",i,err(i));
end

plot(qpt,uk);
hold on;
plot(qpt,uq);
hold on;
plot(log10((1:iter)'),log10(err),'.r');

%2d gauss int,F是函数在2d积分区域上的函数值构成的矩阵
function r=int1d(f)
    r=(18-sqrt(30))/72*(sum(f(1:4:end-3))+sum(f(4:4:end)))+(18+sqrt(30))/72*(sum(f(2:4:end-2))+sum(f(3:4:end-1)));
end
