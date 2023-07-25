
%u是求投影的函数，g(i)是RELU(w(i)x+b(i)),g是行向量是函数空间的一组基,返回cp是基的系数
function cp=projection_u(f,g,n)
    syms x [2,1];
    syms c [2,1];

    gc=g*c;
    dgc1=diff(g,x1)*c;
    dgc2=diff(g,x2)*c;
    eqn=( int(int(gc.*g,x1,0,1),x2,0,1)+int(int(dgc1.*diff(g,x1),x1,0,1),x2,0,1)+int(int(dgc2.*diff(g,x2),x1,0,1),x2,0,1)-int(int(f.*g,x1,0,1),x2,0,1)==zeros(1,n) );
    [A,b]=equationsToMatrix(eqn,c);
    A=eval(A);
    b=eval(b);
    cp=pinv(A)*b;
end