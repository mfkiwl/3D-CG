% Program created by: 
% Ngoc Cuong Nguyen (cuongng@mit.edu) and Carmen Guerra-Garcia (guerrac@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019

function [chi,discharge,Qd,xdischarge,Sq] = Leader2_inception(phi,theta,chi0,Amp0,bias_s,int23,Rf,C,LAPLACE)

%This script iterates on the charge of the aircraft, as the first leader propagates, until the SECOND 
% leader is incepted. The leader inception condition can be based on either a surface
%or a volume integral. Choice of criterion is done through parameter int23 (2=surface, 3=volume). 
%The results are for a given orientation of the ambient electric field (phi,theta), and magnitude of the field
% found for the first inception condition. The second attachment point is also evaluated (xdischarge).

%physical constants
[ ~, ~, Qtp, Qtn , eps0, ~ ] = physical_constants;

dchi  = 0.1e-6*(Rf/0.07)^2; %The Rf factor helps scale the charge increment to take with the size
dchi  = dchi/(1e3*Rf*C);

Critp = Qtp/(eps0*1e3*Rf*Rf);
Critn = Qtn/(eps0*1e3*Rf*Rf);

chi   = chi0;
while (1)

    if int23 == 2
    [Ipmax,imax,Ipmin,imin,S,~] = Leader_S(chi,phi,theta,Amp0,LAPLACE);
    elseif int23 == 3
    [Ipmax,imax,Ipmin,imin,S,~] = Leader_V(chi,phi,theta,Amp0,LAPLACE);
    end

    if  (bias_s>0)&&(Ipmax >= Critp)  % positive corona discharge
        discharge = imax; % location of discharge  
        Qd        = Ipmax;
        xdischarge = LAPLACE.msh.p(LAPLACE.xpoint(discharge),:);
        Sq         = S(imax);
        break;                 
    elseif  (bias_s<0)&&(Ipmin <= -Critn) % negative corona discharge
        discharge = imin; % location of discharge    
        Qd = Ipmin;
        xdischarge = LAPLACE.msh.p(LAPLACE.xpoint(discharge),:);
        Sq         = S(imin);
        break; 
    else
        chi = chi + bias_s*dchi;        
    end    

end

end

