
%u是求投影的函数，g(i)是RELU(w(i)x+b(i)),g是行向量是函数空间的一组基,返回cp是基的系数
function cp=projection_phi(w,b)
    syms x;
    [~,n]=size(w);
    
    for i=1:n
        g(i)=RELU(x*w(i)+b(i),1);
    end
    c=sym("c",[n 1]);
    for i=1:n
        eqn(i)=( int((g*c)*g(i),x,0,1) + int((diff(g)*c)*diff(g(i)),x,0,1) - subs(g(i),x,1) ==0);
    end
    [A,b]=equationsToMatrix(eqn,c);
    A=eval(A);
    b=eval(b);
    cp=A\b;
end