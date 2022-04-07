% Program created by: 
% Ngoc Cuong Nguyen (cuongng@mit.edu) and Carmen Guerra-Garcia (guerrac@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019

function [xdischarge,Idischarge,Amp,I,S,Ip,In,xPOS,xNEG] = Leader1_inception(chi,phi,theta,Amp0,int23, Rf,LAPLACE)

%This script iterates on the amplitude of the ambient field until the first 
% leader is incepted. The leader inception condition can be based on either a surface
%or a volume integral. Choice of criterion is done through parameter int23 (2=surface, 3=volume). 
%The results are for a given orientation of the ambient electric field (phi,theta), and a prescribed net aircraft
%charge (chi). The first attachment point is also evaluated (xdischarge).

% constants
[ ~, ~, Qtp, Qtn , eps0, ~ ] = physical_constants;

Critp = Qtp/(eps0*1e3*Rf*Rf);
Critn = Qtn/(eps0*1e3*Rf*Rf);

A1 = Amp0;
Amp = Amp0; 

% Calculate entry point and ambient electric field for breakdown (Einf [kV/m]): iterate
% on the ambient field until leader is incepted in one extremity of the model
    
while (1)
        
    if int23 == 2
    [Ipmax,imax,Ipmin,imin,S,I] = Leader_S(chi,phi,theta,Amp,LAPLACE);
    elseif int23 == 3
    [Ipmax,imax,Ipmin,imin,S,I] = Leader_V(chi,phi,theta,Amp,LAPLACE);
    end

    if  Ipmax >= Critp  % positive corona discharge
        discharge = imax; % location of discharge               
        break;        
    elseif  Ipmin <= -Critn % negative corona discharge
        discharge = imin; % location of discharge                                
        break;
    else
        % increase Amp
        A1  = Amp; 
        Amp = 2*Amp;        
    end
end

A2 = Amp; % discharge always occurs at A2

if A1==A2     
    Amp = A2/2;
    while (1)  
        
        if int23 == 2
        [Ipmax,imax,Ipmin,imin,S,I] = Leader_S(chi,phi,theta,Amp,LAPLACE);
        elseif int23 == 3
        [Ipmax,imax,Ipmin,imin,S,I] = Leader_V(chi,phi,theta,Amp,LAPLACE);
        end
        
        if  (Ipmax < Critp) && (Ipmin > -Critn)  
            % discharge does not occur at Amp
            A1 = Amp;
            % discharge occur at 2*Amp
            A2 = 2*Amp;             
            break;                
        else
            % decrease Amp            
            Amp = Amp/2;        
        end            
        
        if Amp<1e-6
            break;
        end        
    end           
    
    if Amp<1e-6        
        [Ipmax,imax] = max(I); % maximum charge
        [Ipmin,imin] = min(I); % minimum charge                
        if  Ipmax >= Critp  % positive corona discharge
            discharge = imax; % location of discharge                           
        elseif  Ipmin <= -Critn % negative corona discharge
            discharge = imin; % location of discharge   
        else
            error('something wrong');
        end
        
        xdischarge = LAPLACE.msh.p(LAPLACE.xpoint(discharge),:);
        Idischarge = I(discharge);
        xPOS       = LAPLACE.msh.p(LAPLACE.xpoint(imax),:);
        xNEG       = LAPLACE.msh.p(LAPLACE.xpoint(imin),:);
        Ip         = Ipmax;
        In         = Ipmin;
        return;
    end
end

if abs(A2-2*A1)>1e-10
    error('something wrong');
end

while (1)
    Amp = 0.5*(A1+A2);

    if int23 == 2
    [Ipmax,imax,Ipmin,imin,S,I] = Leader_S(chi,phi,theta,Amp,LAPLACE);
    elseif int23 == 3
    [Ipmax,imax,Ipmin,imin,S,I] = Leader_V(chi,phi,theta,Amp,LAPLACE);
    end
        
    if  (Ipmax >= Critp)  % positive corona discharge
        discharge = imax; % location of discharge                  
        A2 = Amp;              
    elseif  Ipmin <= -Critn % negative corona discharge
        discharge = imin; % location of discharge                        
        A2 = Amp;        
    else        
        A1  = Amp;         
    end    
    if (A2-A1) < 0.01*abs(Amp)
        break;
    end
end

xdischarge = LAPLACE.msh.p(LAPLACE.xpoint(discharge),:);
Idischarge = I(discharge);
xPOS       = LAPLACE.msh.p(LAPLACE.xpoint(imax),:);
xNEG       = LAPLACE.msh.p(LAPLACE.xpoint(imin),:);
Ip         = Ipmax;
In         = Ipmin;

end









