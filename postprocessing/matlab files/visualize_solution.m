% Program created by: 
% Carmen Guerra-Garcia (guerrac@mit.edu) and Ngoc Cuong Nguyen (cuongng@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019
 
% Plotting script

% Plots on the aircraft surface the normal electric field value [kV/m] for a
% given electric field orientation (theta, phi) and magnitude (Amp), 
% and aircraft net charge level (chi)

clear all
close all
clc

%%%%%%%%% THESE ARE THE INPUTS OF THE MODEL  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
                                                                                                %%
% Dimensions of aircraft, geometry (Rf: fuselage radius, C: capacitance)                        %%
Rf  =  3.6;                     % Fuselage radius [m]; for this particular geometry,            %%
                                % the wing span is 18*Rf                                        %%           
                                                                                                %%
LAPLACE=load('falconfine.mat'); % Results from Laplace solver given geometry                    %%
LAPLACE.bft = LAPLACE.bft{1};                                                                   %%
LAPLACE.dgn = LAPLACE.dgn{1};                                                                   %%
                                                                                                %%
[C,~] = Capacitance_calc(Rf,LAPLACE);   % C: capacitance [F] - calculated from CAD geometry     %%
                                                                                                %%
% Model orientation in external field and initial charge                                        %%
chi   = 0;   % Net charge in Coulomb                                                         %%
phi   = 0;       % Yaw angle in degrees (see diagram)                                            %%
theta = 0;      % Pitch angle in degrees (see diagram)                                          %%
Amp   = 30;      % Amplitude of the external electric field in kV/m .                            %%
                                                                                                %%                                                                                 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

chi   = chi/(1e3*Rf*C);
phi   = phi*pi/180;
theta = theta*pi/180;

dx = sin(theta)*cos(phi); 
dy = sin(theta)*sin(phi); 
dz = cos(theta);

nd = LAPLACE.msh.nd;
[npf,~] = size(LAPLACE.msh.perm);
gwfc = LAPLACE.master.gwfc;
shapfc = LAPLACE.master.shapfc;
shapft  = shapfc(:,:,1)';
dshapft = reshape(permute(shapfc(:,:,2:end),[2 3 1]),[npf*(nd-1) npf]);
En = - chi*LAPLACE.Un{1} + Amp*(dx*LAPLACE.Un{2} + dy*LAPLACE.Un{3} + dz*LAPLACE.Un{4});
porder=LAPLACE.msh.porder;
plocal=LAPLACE.msh.plocfc;
tlocal=LAPLACE.msh.tlocfc;
nref = 2;
A0=koornwinder(plocal(:,1:2),porder);
[plocal,tlocal]=uniref(plocal,tlocal,nref);
A=koornwinder(plocal(:,1:2),porder)/A0;
En0=A*En;

[npf,~] = size(LAPLACE.msh.perm);
npln=size(plocal,1);
sz=size(LAPLACE.dgn); if length(sz)==2, sz = [sz,1]; end
dg0=reshape(A*reshape(LAPLACE.dgn,npf,sz(2)*sz(3)),[npln,sz(2),sz(3)]);

nt = size(En0,2);
nodesvis=reshape(permute(dg0,[1,3,2]),[npln*nt,nd]);
tvis=kron(ones(nt,1),tlocal)+kron(npln*(0:nt-1)',0*tlocal+1);
figure(1); clf;
set(axes,'FontSize',16);
patch('vertices',nodesvis,'faces',tvis,'cdata',En0(:), ...
           'facecol','interp','edgec','none');  
hold on;
set(gca,'clim',[min(min(-10*abs(Amp),chi),0) max(max(10*abs(Amp),chi),0)]);
colormap jet;  
axis equal;axis tight;
set(gca,'xtick',[],'ytick',[],'ztick',[])
set(gca,'Visible','off')     
h = quiver3(0,0,1,4*dx,4*dy,4*dz,'k','Linewidth',4);
set(h,'MaxHeadSize',1e2,'AutoScaleFactor',1);
view([-20 20])
colorbar
text(0,0,10,'Electric field [kV/m]','FontSize',16)

%View potential solution for slice in kV

figure(2)

Phi = chi*LAPLACE.UDG{1}(:,1,:) - Amp*(dx*LAPLACE.UDG{2}(:,1,:) + dy*LAPLACE.UDG{3}(:,1,:) + dz*LAPLACE.UDG{4}(:,1,:));
Phi = Phi*Rf;

subplot(2,1,1)
title('Potential [MV]')
[~,~,~] = sliceplot(LAPLACE.msh, Phi.*1e-3, [0 1 0 0], 1);
set(gca,'clim',[min(min(-10*abs(Amp*Rf),chi*Rf),0)*1e-3 max(max(10*abs(Amp*Rf),chi*Rf),0)*1e-3]); 
view(0,0) %XZ

subplot(2,1,2)
[~,~,~] = sliceplot(LAPLACE.msh, Phi*1e-3, [0 0 1 0.8], 1);
set(gca,'clim',[min(min(-abs(10*Amp*Rf),chi*Rf),0)*1e-3 max(max(abs(10*Amp*Rf),chi*Rf),0)*1e-3]); 
view(0,90) %XY

