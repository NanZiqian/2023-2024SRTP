function r=RELU(x,k)

r=piecewise(x<=0,0,x>0,x^k);

