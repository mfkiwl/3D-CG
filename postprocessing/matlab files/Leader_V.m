% Program created by: 
% Ngoc Cuong Nguyen (cuongng@mit.edu) and Carmen Guerra-Garcia (guerrac@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019

function [Ipmax,imax,Ipmin,imin,S,I] = Leader_V(chi,phi,theta,Amp,LAPLACE)

%This script evaluates the LEADER inception criterion using a VOLUME integral. 

% constants
[  Etp, Etn, ~, ~ , ~, ~ ] = physical_constants;

% faces contain the extreme points of the aircraft
[~,ixface] = ismember(LAPLACE.xface,LAPLACE.bft);

% spherical coordinate 
dx = sin(theta)*cos(phi); 
dy = sin(theta)*sin(phi); 
dz = cos(theta);

% get shape functions to compute the integrals
nd = LAPLACE.msh.nd;
ne = LAPLACE.msh.ne;
npv = LAPLACE.master.npv;
ngv = LAPLACE.master.ngv;

% centroid of the elements
c = 0.25*(LAPLACE.msh.p(LAPLACE.msh.t(:,1),:)+LAPLACE.msh.p(LAPLACE.msh.t(:,2),:)+LAPLACE.msh.p(LAPLACE.msh.t(:,3),:)+LAPLACE.msh.p(LAPLACE.msh.t(:,4),:));

% extreme points of the aircraft
nxp = length(LAPLACE.xpoint);
phip = zeros(ne,nxp);
phin = zeros(ne,nxp);
dc = zeros(ne,nxp);
ec = zeros(ne,nxp);
for j = 1:nxp % loop over each extreme point  
    px = LAPLACE.msh.p(LAPLACE.xpoint(j),:);  % get coordinates of the extreme point j
    % compute the distance from the elements to the extreme point
    dc(:,j) = sqrt((px(1) - c(:,1)).^2 + (px(2) - c(:,2)).^2 + (px(3) - c(:,3)).^2);
    [~,ec(:,j)] = sort(dc(:,j));
    phip(:,j) = chi - Etp*dc(:,j);
    phin(:,j) = chi + Etn*dc(:,j);
end

% values of the shape function at the centroid points
shapc = mkshape(LAPLACE.master.porder,LAPLACE.master.plocvl,[1/4 1/4 1/4],LAPLACE.msh.elemtype);
shapc = shapc(:,:,1)';

gwvl = LAPLACE.master.gwvl;
shapvt = reshape(permute(LAPLACE.master.shapvl(:,:,1),[2 3 1]),[ngv npv]);
dshapvt = reshape(permute(LAPLACE.master.shapvt(:,:,2:nd+1),[1 3 2]),[ngv*nd npv]);

    % calculate the average electric field for all surface elements
    Enavg = -chi*LAPLACE.Unavg{1} + Amp*(dx*LAPLACE.Unavg{2} + dy*LAPLACE.Unavg{3} + dz*LAPLACE.Unavg{4});    
    
    Phi = chi*LAPLACE.UDG{1}(:,1,:) - Amp*(dx*LAPLACE.UDG{2}(:,1,:) + dy*LAPLACE.UDG{3}(:,1,:) + dz*LAPLACE.UDG{4}(:,1,:));
    %sliceplot(msh, Phi, [0 1 0 0], 1);    
    
    % calculate the exterior potential for all elements    
    phie = shapc*reshape(Phi,[npv ne]);    
    phie = phie';
            
    
    S = zeros(nxp,1);
    I = zeros(nxp,1);
    for j = 1:1:nxp % loop over each extreme point  
        ixf = ixface(j);  % get the element contains the extreme point j                
        if Enavg(ixf) > 0 % if it is a positive extreme                
            in = find(phie < phip(:,j));    
            sig = Etp;
        elseif Enavg(ixf) < 0            
            in = find(phie > phin(:,j));                                    
            sig = -Etn;
        else
            in = [];
            sig = 0;
        end      
        for k = 1:length(in)
            % element i
            i = in(k); 
            
            % get nodes of element i
            pn = LAPLACE.msh.dgnodes(:,:,i);
            pg = reshape(shapvt*pn,[ngv nd]);           
            
            % compute Kphi
            rg = sqrt(pg(:,1).^2 + pg(:,2).^2 + pg(:,3).^2);
            Kp = 2./rg;

            % compute the mapping jacobian
            Jg = dshapvt*pn(:,1:nd);
            Jg = reshape(Jg,[ngv nd nd]);    
            jac = Jg(:,1,1).*Jg(:,2,2).*Jg(:,3,3) - Jg(:,1,1).*Jg(:,3,2).*Jg(:,2,3)+ ...
                  Jg(:,2,1).*Jg(:,3,2).*Jg(:,1,3) - Jg(:,2,1).*Jg(:,1,2).*Jg(:,3,3)+ ...
                  Jg(:,3,1).*Jg(:,1,2).*Jg(:,2,3) - Jg(:,3,1).*Jg(:,2,2).*Jg(:,1,3);            

            % compute the volume
            S(j) = S(j) + sum(gwvl.*jac);
            
            % compute the charge at the extreme point 
            I(j) = I(j) + sig*sum(gwvl.*Kp.*jac);           
        end        
    
    [Ipmax,imax] = max(I); % maximum charge
    [Ipmin,imin] = min(I); % minimum charge
    
    end
end



