% Program created by: 
% Ngoc Cuong Nguyen (cuongng@mit.edu) and Carmen Guerra-Garcia (guerrac@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019

function [ Etp, Etn, Qtp, Qtn, eps0, d_R] = physical_constants

% Thresholds for leader inception (chosen at standard conditions) and universal constants

% Physical constants
eps0= 8.85e-12;    %vacuum permittivity [F/m]

Etp = 450;         %stability field of positive corona [kV/m]
Etn = 750;         %stability field of negative corona [kV/m]
Qtp = 1e-6;        %Critical corona charge for positive leader inception [C]
Qtn = 4*Qtp;       %Critical corona charge for negative leader inception [C]

d_R = 0.75;        %Artificial constraint to limit extension of surface 
                   %integration for corona charge calculation [],
                   %nondimensionalized with Rf
end

