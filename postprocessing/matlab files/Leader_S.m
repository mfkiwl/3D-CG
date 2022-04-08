% Program created by: 
% Ngoc Cuong Nguyen (cuongng@mit.edu) and Carmen Guerra-Garcia (guerrac@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019

function [Ipmax,imax,Ipmin,imin,S,I] = Leader_S(chi,phi,theta,Amp,LAPLACE)

%This script evaluates the LEADER inception criterion using a SURFACE integral. 

% constants
[  Etp, Etn, ~, ~ , ~, dmax ] = physical_constants;

% faces contain the extreme points of the aircraft
[~,ixface] = ismember(LAPLACE.xface,LAPLACE.bft);

% spherical coordinate 
dx = sin(theta)*cos(phi); 
dy = sin(theta)*sin(phi); 
dz = cos(theta);

% get shape functions to compute the integrals
nd = LAPLACE.msh.nd;
[npf,~] = size(LAPLACE.msh.perm);
gwfc = LAPLACE.master.gwfc;
shapfc = LAPLACE.master.shapfc;
shapft  = shapfc(:,:,1)';
dshapft = reshape(permute(shapfc(:,:,2:end),[2 3 1]),[npf*(nd-1) npf]);
    
    % calculate the average electric field for all surface elements
    Enavg = -chi*LAPLACE.Unavg{1} + Amp*(dx*LAPLACE.Unavg{2} + dy*LAPLACE.Unavg{3} + dz*LAPLACE.Unavg{4});

    % get elements that are bigger than the positive threshold 
    inp = find(Enavg > Etp); 
    bfp = LAPLACE.bft(inp);    
    tep = LAPLACE.msh.f(bfp,1:end-2); 
    % compute the center coordinates of these elements 
    pcp = zeros(length(inp),nd);
    for i = 1:length(inp)
        pcp(i,:) = mean(LAPLACE.msh.p(tep(i,:),:),1);
    end

    % get elements that are smaller than the negative threshold 
    inn = find(Enavg < -Etn);
    bfn = LAPLACE.bft(inn);    
    ten = LAPLACE.msh.f(bfn,1:end-2); 
    % compute the center coordinates of these elements 
    pcn = zeros(length(inn),nd);
    for i = 1:length(inn)
        pcn(i,:) = mean(LAPLACE.msh.p(ten(i,:),:),1);
    end

    nxp = length(LAPLACE.xpoint);
    S = zeros(nxp,1);
    I = zeros(nxp,1);
    for j = 1:nxp % loop over each extreme point              
        px = LAPLACE.msh.p(LAPLACE.xpoint(j),:);  % get coordinates of the extreme point j
        ixf = ixface(j);  % get the element contains the extreme point j
        in = [];
        if Enavg(ixf) > Etp % if it is a positive extreme
            % compute the distance from the extreme point to the neigboring elements
            dd = sqrt((pcp(:,1)-px(1)).^2 + (pcp(:,2)-px(2)).^2 + (pcp(:,3)-px(3)).^2);
            % only select elements that are close enough to the extreme point 
            di = dd<dmax;                
            % list of elements in the neighborhood of the extreme point j
            in = inp(di); 
        elseif Enavg(ixf) <- Etn % if it is a negative extreme
            % compute the distance from the extreme point to the neigboring elements
            dd = sqrt((pcn(:,1)-px(1)).^2 + (pcn(:,2)-px(2)).^2 + (pcn(:,3)-px(3)).^2);
            % only select elements that are close enough to the extreme point 
            di = dd<dmax;        
            % list of elements in the neighborhood of the extreme point j
            in = inn(di);             
        end    
        for k = 1:length(in)        
            % element i
            i = in(k); 
            % get nodes of element i
            p = LAPLACE.dgn(:,:,i);
            % get the electric field on the element i
            En = - chi*LAPLACE.Un{1}(:,i) + Amp*(dx*LAPLACE.Un{2}(:,i) + dy*LAPLACE.Un{3}(:,i) + dz*LAPLACE.Un{4}(:,i));
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
            S(j) = S(j) + sum(gwfc.*jacf);
            % compute the charge at the extreme point 
            I(j) = I(j) + sum(gwfc.*Eng.*jacf);           
        end    
    
    [Ipmax,imax] = max(I); % maximum charge
    [Ipmin,imin] = min(I); % minimum charge

end


end




