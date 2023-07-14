
% < u,v >_H=(u,v)+(u',v')
% u,v are symbolic functions, integrate u,v over x from a to b
function [result] = Product_H(u,v,a,b)
    result = int(u*v,x,a,b)+int( diff(u)*diff(v),x,a,b);
end

