
%u是求投影的函数，g(i)是RELU(w(i)x+b(i)),g是行向量是函数空间的一组基,返回cp是基的系数
function cp=projection(u,w,b)
    syms x;
    [~,n]=size(w);
    
    g=arrayfun(@(x) RELU(x,2),w.*x+b);
    c=sym("c",[n 1]);
%     for i=1:n
%         eqn(i)=( XGaussIntegral((u-g*c)*g(i),0,1,100) +  XGaussIntegral((diff(u)-diff(g)*c)*diff(g(i)),0,1,100)   ==0);
%     end
    u_gc=u-g*c;
    du_dgc=diff(u)-diff(g)*c;
    eqn=( int(u_gc.*g,x,0,1)+int(du_dgc.*diff(g),x,0,1)==zeros(1,n) );
    [A,b]=equationsToMatrix(eqn,c);
    A=eval(A);
    b=eval(b);
    cp=A\b;
end