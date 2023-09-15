%% solve 2d equation -Δu+u=f on[0 1]×[0 1] with OGA iterations
function [uk,err_L2,err_H1]=OGA_2D_ori(BASE_SIZE,N,ht,hb)
%source function -Δu+u=f u=cos(2*pi*x1)*cos(2*pi*x2)
f=@(x1,x2) (1+8*pi^2)*cos(2*pi*x1)*cos(2*pi*x2);
u=@(x1,x2) cos(2*pi*x1)*cos(2*pi*x2);%方程精确解用于误差估计
du1=@(x1,x2) -2*pi*sin(2*pi*x1)*cos(2*pi*x2);
du2=@(x1,x2) -2*pi*cos(2*pi*x1)*sin(2*pi*x2);
%BASE_SIZE=100;
iter=BASE_SIZE;%迭代次数
nr=2;%ReLU的阶数

%N=50;%积分区间划分数量
h=1/N;%区间长度
H=h^2;
node=(0:h:1);
T=[1/2-sqrt(15)/10 1/2 1/2+sqrt(15)/10];%高斯积分的区间划分
L=3*N;%积分点数量
qpt=zeros(1,L);%quadrate point 高斯积分点向量
qpt(1:3:end-2)=node(1:end-1)+T(1)*h;
qpt(2:3:end-1)=node(1:end-1)+T(2)*h;
qpt(3:3:end)=node(1:end-1)+T(3)*h;
qar=repmat(qpt,L,1);%积分点矩阵，类似于meshgrid生成的矩阵

%ht=0.9;%角度步长，用于w1,w2的离散|w|=1
%hb=0.1;%b的步长
delta=0.0001;%用于数值求导
nd=(90/ht+1)*(4/hb+1);%字典中函数数量
g=zeros(L,L,nd);%离散化的字典 g=ReLU^2(w1x1+w2x2+b)
dg1=zeros(L,L,nd);%g对x1求导的导函数
dg2=zeros(L,L,nd);%g对x2求导的导函数
j=1;
for theta=(0:ht:90)
    for b=(-2:hb:2)
        csb=cos(theta)*qar'+sin(theta)*qar+b;
        g(:,:,j)=max(0,csb).^nr;%g=ReLU^2(w1x1+w2x2+b)
        dg1(:,:,j)=(max(0,csb+cos(theta)*(delta)).^nr-max(0,csb-cos(theta)*(delta)).^nr)/(2*delta);
        dg2(:,:,j)=(max(0,csb+sin(theta)*(delta)).^nr-max(0,csb-sin(theta)*(delta)).^nr)/(2*delta);
        j=j+1;
    end
end

fqpt=f(qpt',qpt);
uq=u(qpt',qpt);
duq1=du1(qpt',qpt);
duq2=du2(qpt',qpt);
uk=zeros(L,L);%数值解
duk1=zeros(L,L);%uk对x1求导的导函数
duk2=zeros(L,L);%uk对x2求导的导函数
id=zeros(iter,1);%标记字典中组成uk的基的索引
err_L2=zeros(iter,1);err_H1=zeros(iter,1);
argmax=zeros(nd,1);%存储|<g,u-uk>|的值
A = zeros(iter,iter); rhs = zeros(iter,1);%求解投影的线性方程组系数矩阵与常数项矩阵
for i=1:iter 
    for j=1:nd
        argmax(j)=int2d(g(:,:,j).*(uk-fqpt)+dg1(:,:,j).*duk1+dg2(:,:,j).*duk2)*H;
    end
    [~,id(i)] = max(abs(argmax));
    for j = 1:i
        A(j,i) = int2d(g(:,:,id(j)).*g(:,:,id(i))+dg1(:,:,id(j)).*dg1(:,:,id(i))+dg2(:,:,id(j)).*dg2(:,:,id(i)))*H;
        A(i,j) = A(j,i);
    end
    rhs(i) = int2d(g(:,:,id(i)).*fqpt)*H;
    C = lsqminnorm(A(1:i,1:i),rhs(1:i));
    %C=A(1:i,1:i)\rhs(1:i);
    uk=zeros(L,L);%数值解
    duk1=zeros(L,L);%uk对x1求导的导函数
    duk2=zeros(L,L);%uk对x2求导的导函数
    for j = 1:i
        uk=uk+g(:,:,id(j))*C(j);
        duk1=duk1+dg1(:,:,id(j))*C(j);
        duk2=duk2+dg2(:,:,id(j))*C(j);
    end

    r=uq-uk;
    dr1=duq1-duk1;
    dr2=duq2-duk2;
    err_L2(i) = sqrt(int2d(r.^2)*H);
    err_H1(i) = sqrt(int2d(r.^2+dr1.^2+dr2.^2)*H);
    %fprintf("Step %d, error_L2 is %f\n",i,err(i));
end

%subplot(1,2,1);
%mesh(qpt',qpt,uk);
%hold on;
%mesh(qpt',qpt,uq);
%subplot(1,2,2);
%plot(log10((1:iter)'),log10(err),'.r');

end

%2d gauss int,F是函数在2d积分区域上的函数值构成的矩阵
function r=int2d(F)
    f=5/18*sum(F(:,1:3:end-2),2)+4/9*sum(F(:,2:3:end-1),2)+5/18*sum(F(:,3:3:end),2);
    r=5/18*sum(f(1:3:end-2))+4/9*sum(f(2:3:end-1))+5/18*sum(f(3:3:end));
end




