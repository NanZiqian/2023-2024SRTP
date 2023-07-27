
%u是求投影的函数，g(i)是RELU(w(i)x+b(i)),g是行向量是函数空间的一组基,返回cp是基的系数
function un=projection_u(f,g,n)
    syms x;
    c=sym("c",[n 1]);
    
    gc=g*c;
    dgc=diff(g)*c;
    eqn=( int(gc.*g,x,0,1)+int(dgc.*diff(g),x,0,1)-int(f.*g,x,0,1)==zeros(1,n) );
    [A,b]=equationsToMatrix(eqn,c);
    A=eval(A);
    b=eval(b);
    cp=pinv(A)*b;
    un=g*cp;
end