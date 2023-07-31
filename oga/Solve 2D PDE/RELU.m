function r=RELU(x,k)

r=arrayfun(@(a) piecewise(a<=0,0,a>0,a^k),x);
