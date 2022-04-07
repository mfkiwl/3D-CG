% Program created by: 
% Ngoc Cuong Nguyen (cuongng@mit.edu) and Carmen Guerra-Garcia (guerrac@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019

function [ ind_p ] = att_point_index( xpoint, msh, x_cor )

% This script assigns a number 1:11 to the 11 possible attachment points
% considered: see diagram


rr_p(1:11)       = nan;

for i=1:11
    
    rr_p(i) = msh.p(xpoint(i),3) + msh.p(xpoint(i),2) + msh.p(xpoint(i),1);
    
end



        rr = x_cor(1) + x_cor(2) + x_cor(3);
        
        for i=1:11
           
            if abs(rr-rr_p(i))<1e-4
                
                ind_p = i;
                
            end
            
        end
        
 


end

