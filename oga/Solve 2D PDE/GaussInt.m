
%分成N个区域，有N+1个点
function [result] = GaussInt(f,x,low,up,N)

    delta=(up-low)/N;

    %xN=zeros(N+1,1);%离散积分点
    %for i=1:N+1
    %    xN(i)=low+(i-1)*delta;
    %end
    xN=linspace(0,N,N+1)'*delta+low*ones(N+1,1);
    syms t;
    T = [-0.9061798,-0.5384693,0.0000000,0.5384693,0.9061798];
    Ak = [0.2369269,0.4786287,0.5688889,0.4786287,0.2369269]';

    X=(xN(2:N+1)*(1+t)+xN(1:N)*(1-t))/2;
    XN = subs(X,t,T);
    fN = subs(f,x,XN);
    r=ones(1,N)*fN;
    result = r*Ak*delta/2;
    
end

