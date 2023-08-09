
%分成N个区域，有N+1个点
function [result] = XGaussIntegral(f,a1,b1,N)

    xN=zeros(N+1,1);%离散积分点
    for i=1:N+1
        xN(i)=a1+(i-1)/N*(b1-a1);
    end
    
    syms x;
    syms t;
    f=subs(f,x);
    result=0;
    T = [-0.9061798,-0.5384693,0.0000000,0.5384693,0.9061798];
    Ak = [0.2369269,0.4786287,0.5688889,0.4786287,0.2369269];

    for j=1:N
        a=xN(j);
        b=xN(j+1);
        r = 0;
        X = (b-a)*t/2+(b+a)/2;
        for i=1:5
            XN = subs(X,t,T(i));
            fN = subs(f,x,XN);
            r = r+Ak(i)*fN;
        end
        r = r*(b-a)/2;
        result=result+r;
    end
end

