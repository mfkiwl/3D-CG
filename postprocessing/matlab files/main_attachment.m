% Program created by: 
% Carmen Guerra-Garcia (guerrac@mit.edu) and Ngoc Cuong Nguyen (cuongng@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019


% MAIN SCRIPT
%
% This script evaluates for any orientation of the given aircraft
% geometry (and any net charge) the ambient electric field amplitude for breakdown (Einf [kV/m]) 
% as well as the entry and exit points of the discharge
%
% Main outputs: Einf [kV/m], ind_L1 (index of entry point), ind_L2 (index of
% exit point) - see diagram to relate indeces to points on aircraft

clear all
close all
clc

%%%%%%%%% THESE ARE THE INPUTS OF THE MODEL  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
                                                                                                %%
% Dimensions of aircraft, geometry (Rf: fuselage radius, C: capacitance)                        %%
Rf  =  1.25;                     % Fuselage radius [m]; for this particular geometry,            %%
                                % the wing span is 18*Rf                                        %%           
                                                                                                %%
LAPLACE=load('falconfine.mat'); % Results from Laplace solver given geometry                    %%
LAPLACE.bft = LAPLACE.bft{1};                                                                   %%
LAPLACE.dgn = LAPLACE.dgn{1};                                                                   %%
                                                                                                %%
[C,~] = Capacitance_calc(Rf,LAPLACE);   % C: capacitance [F] - calculated from CAD geometry     %%
                                                                                                %%
% Model orientation in external field and initial charge                                        %%
chi   = 0;      % Net charge in Coulomb                                                         %%
phi   = 180;      % Yaw angle in degrees (see diagram)                                            %%
theta = 50;      % Pitch angle in degrees (see diagram)                                          %%
                                                                                                %%
% Choice of leader criterion based on surface charge (int23 = 2) or volume charge (int23 = 3)   %%      
int23   =3;                                                                                     %%
                                                                                                %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

chi   = chi/(1e3*Rf*C);
phi   = phi*pi/180;
theta = theta*pi/180;

Qac_init   = chi*1e3*Rf*C; %[C] This is the net charge of the aircraft 

% Parameters: thresholds for leader inception and universal constants
[  Etp, Etn, Qtp, Qtn , eps0, dmax ] = physical_constants;

% Calculate entry point and ambient electric field for breakdown (Einf [kV/m]): iterate
% on the ambient field until leader is incepted in one extremity of the model

Amp0 = Etp/10; % Ambient field iteration initial value

[xdischarge,Idischarge,Einf,I,S,Ip,In,xPOS,xNEG]= Leader1_inception(chi,phi,theta,Amp0,int23, Rf,LAPLACE);

if int23 == 2 % Use surface integral for leader inception criterion
S_L1       = S*Rf*Rf; %[m2]
elseif int23 == 3 % Use volume integral for leader inception criterion
S_L1       = S*Rf*Rf*Rf; %[m3]
end

Qc_L1      = eps0*Rf*Rf*Idischarge*1e3*1e6; %[micro-C] Corona charge at 1st leader inception
x_L1       = xdischarge; % Attachment point

% Index of entry point
[ ind_L1 ] = att_point_index( LAPLACE.xpoint, LAPLACE.msh, x_L1 );
[ dp ] = select_direction(ind_L1);

S_L1 = S_L1(ind_L1);

% Double check corona inception
[~ ,~ ,~ ,~ ,~ ,~ ,Q1D_L1,Qlim_L1] = Corona_line(chi,phi,theta,Einf,x_L1,dp,LAPLACE.UDG,LAPLACE.master,LAPLACE.msh,Rf);

 % Is entry point positive or negative?
bias_s = sign(-Idischarge); %if positive leader incepted, body gets negatively charged

% Calculate exit point: iterate on the vehicle charge as the first leader propagates until
% the second leader is incepted in one extremity of the model

[chi_L2,ind_L2,Idischarge2,x_L2,Sq] = Leader2_inception(phi,theta,chi,Einf,bias_s,int23,Rf,C,LAPLACE);
 
if int23 == 2 % Use surface integral for leader inception criterion
S_L2       = Sq*Rf*Rf; %[m2]
elseif int23 == 3 % Use volume integral for leader inception criterion
S_L2       = Sq*Rf*Rf*Rf; %[m3]
end

Qc_L2      = eps0*Rf*Rf*Idischarge2*1e3*1e6; %[micro-C] Corona charge at 2nd leader inception
Qac        = chi_L2*1e3*Rf*C; %[C] Aircraft charge at 2nd leader inception (due to propagation of 1st leader)

[ dp ] = select_direction(ind_L2);

% Double check corona inception
[~ ,~ ,~ ,~ ,~ ,~ ,Q1D_L2,Qlim_L2] = Corona_line(chi_L2,phi,theta,Einf,x_L2,dp,LAPLACE.UDG,LAPLACE.master,LAPLACE.msh,Rf);

% Display results
Qac_mC = Qac*1e3; %Aircraft charge at 2nd leader inception in mC
V_acMV   = Qac/C.*1e-6; %Aircraft potential at 2nd leader inception in MV
Einf_kVm = Einf; %Breakdown value of the electric field to trigger leaders

T    = table(Einf_kVm, ind_L1, ind_L2);
T.Properties.VariableUnits = {'kV/m' '' ''};

sign_L1 = -bias_s;
sign_L2 = bias_s;

T1 = table(ind_L1,sign_L1, Qc_L1, S_L1, Q1D_L1);
T1.Properties.VariableUnits = {'' '' 'micro-C' 'm^2/m^3' 'N'};

T2 = table(ind_L2,sign_L2, Qc_L2, S_L2, Q1D_L2);
T2.Properties.VariableUnits = {'' '' 'micro-C' 'm^2/m^3' 'N'};

summary(T)

