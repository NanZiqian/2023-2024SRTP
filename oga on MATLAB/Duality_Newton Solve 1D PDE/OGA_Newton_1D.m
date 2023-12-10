%% solve 2d equation -Δu+u=f on[0 1]×[0 1] with OGA iterations
clear;
%source function -u''+u=f,f=(1+pi^2)*cos(pi*x);
f=@(x) (1+pi^2)*cos(pi*x); 
u=@(x) cos(pi*x);
iter=100;%迭代次数

N=300;%积分区间划分数量
h=1/N;%区间长度
node=(0:h:1);
T=0.5*[1-sqrt((15+2*sqrt(30))/35) 1-sqrt((15-2*sqrt(30))/35) 1+sqrt((15-2*sqrt(30))/35) 1+sqrt((15+2*sqrt(30))/35)];%高斯积分的区间划分
L=4*N;%积分点数量
qpt=zeros(L,1);%quadrate point 高斯积分点向量
qpt(1:4:end-3)=node(1:end-1)+T(1)*h;
qpt(2:4:end-2)=node(1:end-1)+T(2)*h;
qpt(3:4:end-1)=node(1:end-1)+T(3)*h;
qpt(4:4:end)=node(1:end-1)+T(4)*h;

hb=1e-2;%b的步长
delta=1e-6;%用于数值求导
nd=2*(4/hb+1);%字典中函数数量
g=zeros(L,nd);%离散化的字典 g=ReLU^2(wx+b)
dg=zeros(L,nd);%g对x求导的导函数
j=1;
for w=[-1,1]
    for b=(-2:hb:2)
        wqb=w*qpt+b;
        g(:,j)=max(0,wqb).^2;%g=ReLU^2(wx+b)
        dg(:,j)=(max(0,wqb+w*(delta)).^2-max(0,wqb-w*(delta)).^2)/(2*delta);
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


G=@(w,b,x) max(0,w*x+b).^2;
dG_x=@(w,b,x) (G(w,b,x+delta)-G(w,b,x-delta))/(2*delta);
F=@(w,b,x) -1/2*(int1d(G(w,b,x).*(uk-fqpt)+dG_x(w,b,x).*du)*h)^2;
dF_b=@(w,b,x) (F(w,b+delta,x)-F(w,b-delta,x))/(2*delta);
dF_bb=@(w,b,x) (dF_b(w,b+delta,x)-dF_b(w,b-delta,x))/(2*delta);

for i=1:iter 
    for j=1:nd
        argmax(j)=int1d(g(:,j).*(uk-fqpt)+dg(:,j).*du)*h;
    end
    [~,id(i)] = max(abs(argmax));

    %牛顿迭代
    if id(i)<=nd/2
        w0=-1;
        b0=(id(i)-1)*hb-2;
    else
        w0=1;
        b0=(id(i)-nd/2-1)*hb-2;
    end
    while abs(dF_b(w0,b0,qpt))>1e-14
        b0=b0-0.001*dF_b(w0,b0,qpt)/dF_bb(w0,b0,qpt);
    end
    if F(w0,b0,qpt)<-1/2*argmax(id(i))^2
        g(:,id(i))=G(w0,b0,qpt);
        dg(:,id(i))=dG_x(w0,b0,qpt);
    end
    
    for j = 1:i
        A(j,i) = int1d(g(:,id(j)).*g(:,id(i))+dg(:,id(j)).*dg(:,id(i)))*h;
        A(i,j) = A(j,i);
    end
    rhs(i) = int1d(g(:,id(i)).*fqpt)*h;
    C = lsqminnorm(A(1:i,1:i),rhs(1:i));
    %C=A(1:i,1:i)\rhs(1:i);
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
