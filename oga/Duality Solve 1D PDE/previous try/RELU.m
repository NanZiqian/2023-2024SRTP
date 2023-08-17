
% function sigma
function r=RELU(x)
    k=2;
    r=piecewise(x<=0,0,x>0,x^k);
end
