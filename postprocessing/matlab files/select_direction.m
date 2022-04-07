% Program created by: 
% Ngoc Cuong Nguyen (cuongng@mit.edu) and Carmen Guerra-Garcia (guerrac@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019

function [dp] = select_direction(point)

% This script gives the (approximate) normal direction to the surface for
% each possible attachment point. This is the direction of the local
% electric field taken to evaluate the line integral in Corona_line

        if point == 1
            dp = [0 -1 0];
        end
        if point == 2
            dp = [0 -1 0];
        end
        if point == 3
            dp = [0 0 1];
        end
        if point == 4
            dp = [0 0 1];
        end
        if point == 5
            dp = [0 -1 0];
        end
        if point == 6
            dp = [0 -1 0];
        end
        if point == 7
            dp = [0 1 0];
        end
        if point == 8
            dp = [0 1 0];
        end
        if point == 9
            dp = [0 1 0];
        end
        if point == 10
            dp = [0 1 0];
        end
        if point == 11
            dp = [-1 0 0];
        end
        
end