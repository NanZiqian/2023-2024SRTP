

syms x;
F = x^2;

tic
for i = 1: 100
    temp = double(int(F,x,0,1));
end
toc
error_int = temp - 1/3

tic
for i = 1: 100
    temp = double(GaussInt(F,x,0,1,100));
end
toc
error_GaussInt = temp - 1/3