
%u是求投影的函数，g(i)是RELU(w(i)x+b(i)),g是行向量是函数空间的一组基,返回cp是基的系数
function phin=projection_phi(g,n)
    syms x;
    c=sym("c",[n 1]);

    gc=g*c;
    dgc=diff(g)*c;
    eqn=( int(gc.*g,x,0,1)+int(dgc.*diff(g),x,0,1)-subs(g,x,0)==zeros(1,n) );
    [A,b]=equationsToMatrix(eqn,c);
    A=eval(A);
    b=eval(b);
    cp=pinv(A)*b;
    phin=g*cp;
end