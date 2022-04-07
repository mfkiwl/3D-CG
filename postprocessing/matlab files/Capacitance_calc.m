% Program created by: 
% Ngoc Cuong Nguyen (cuongng@mit.edu) and Carmen Guerra-Garcia (guerrac@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019

function [C,S] = Capacitance_calc(Rf,LAPLACE)

% This script calculates the self-capacitance of the aircraft, C [F] and the
% total wetted area, S [m2]. The scaling length (radius of the fuselage Rf
% [m]) needs to be provided externally since the solution to the Laplace solver is
% non-dimensional (LAPLACE)

[ ~, ~, ~, ~, eps0, ~] = physical_constants;

% get shape functions to compute the integrals
nd = LAPLACE.msh.nd;
[npf,~] = size(LAPLACE.msh.perm);
gwfc = LAPLACE.master.gwfc;
shapfc = LAPLACE.master.shapfc;
shapft  = shapfc(:,:,1)';
dshapft = reshape(permute(shapfc(:,:,2:end),[2 3 1]),[npf*(nd-1) npf]);

% calculate the average electric field for all surface elements
Enavg = - LAPLACE.Unavg{1} ;

% get ALL elements
in = length(Enavg); 
bfp = LAPLACE.bft;    
tep = LAPLACE.msh.f(bfp,1:end-2); 
    
% compute the center coordinates of these elements 
pcp = zeros(length(in),nd);
for i = 1:length(in)
    pcp(i,:) = mean(LAPLACE.msh.p(tep(i,:),:),1);
end

S = 0;
I = 0;
   
        for i = 1:in        
            % get nodes of element i
            p = LAPLACE.dgn(:,:,i);
            % get the electric field on the element i
            En = - LAPLACE.Un{1}(:,i);
            % compute the electric field at the Gauss points
            Eng = shapft*En;                        
            % compute the mapping from the element i to the reference element
            dpg = permute(reshape(dshapft*p,[npf nd-1 nd]),[1 3 2]);     
            % compute the normal vector
            nx = reshape(dpg(:,2,1).*dpg(:,3,2) - dpg(:,3,1).*dpg(:,2,2),[npf 1]);
            ny = reshape(dpg(:,3,1).*dpg(:,1,2) - dpg(:,1,1).*dpg(:,3,2),[npf 1]);
            nz = reshape(dpg(:,1,1).*dpg(:,2,2) - dpg(:,2,1).*dpg(:,1,2),[npf 1]);            
            % compute the mapping jacobian
            jacf = sqrt(nx.^2+ny.^2+nz.^2);   
            % compute the area
            S = S + sum(gwfc.*jacf);
            % compute the charge 
            I = I + sum(gwfc.*Eng.*jacf);           
        end    
        
        C = I*eps0*Rf;
        S = S*Rf*Rf;

end