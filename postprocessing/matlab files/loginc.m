% Program created by: 
% Ngoc Cuong Nguyen (cuongng@mit.edu) and Carmen Guerra-Garcia (guerrac@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019

function y=loginc(x,alpha)
a=min(x(:));b=max(x(:));
y = a + (b-a)*(exp(alpha*(x-a)/(b-a))-1)/(exp(alpha)-1);
