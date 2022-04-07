% Program created by: 
% Ngoc Cuong Nguyen (cuongng@mit.edu) and Carmen Guerra-Garcia (guerrac@mit.edu) 
% @MIT AeroAstro under Boeing contract 2016-2019

function [x,Ex,y,Ey,ALPHAy,ETAy,Q,IND_LIM] = Corona_line(chi,phi,theta,Amp0,xp,dp,UDG,master,msh,Rf)

% For a given point (xp) and direction (dp), as well as ambient electric field 
% orientation (phi,theta) and amplitude (Amp0) and net aircraft charge (chi)
% evaluate a line integral to check for corona inception criterion satisfaction
% Qc = exp(\int( alpha -eta )dx). CARE: uses ionization and attachment
% coefficients defined in (swarmair.mat) given for air at atmospheric
% pressure and ambient temperature, will need to revisit these coefficients
% when including altitude effects

% spherical coordinates 
dx = sin(theta)*cos(phi); 
dy = sin(theta)*sin(phi); 
dz = cos(theta);

% electric field
E = chi*UDG{1}(:,2:4,:) - Amp0*(dx*UDG{2}(:,2:4,:) + dy*UDG{3}(:,2:4,:) + dz*UDG{4}(:,2:4,:));
% amplitude
E = reshape(sqrt(E(:,1,:).^2+E(:,2,:).^2+E(:,3,:).^2),[master.npv msh.ne]);

% center point of each element
ne = msh.ne;
xm = zeros(ne,3);
for i = 1:ne
    xm(i,:) = mean(msh.p(msh.t(i,:),:));
end

N = 401;
nref = 1;
dmax = 1;
np = size(xp,1);
for i = 1:np 
    xi = xp(i,:);
    di = dp(i,:);        
    
    % distance from xm to the line xi + t*di
    s = [xm(:,1)-xi(1) xm(:,2)-xi(2) xm(:,3)-xi(3)];
    t = s(:,1)*di(1)+s(:,2)*di(2)+s(:,3)*di(3);
    t = max(t,0);
    s = [xi(1)+t*di(1)-xm(:,1) xi(2)+t*di(2)-xm(:,2) xi(3)+t*di(3)-xm(:,3)];
    d = sqrt(s(:,1).^2+s(:,2).^2+s(:,3).^2);
    
    % make N points from xi to xi+tmax*di
    xmax = 0.1;%max(t(:));   
    x = loginc(linspace(0,xmax,N)',1);
    p = [xi(1)+x*di(1) xi(2)+x*di(2) xi(3)+x*di(3)];
    Ex = 0*p(:,1);
    
    % find elements that contain the line
    inde = find(d<dmax);    
    
    % get scalar field and coordinate on those elements
    udg = E(:,inde);
    pdg = permute(msh.dgnodes(:,1:3,inde),[1 3 2]);

    % subdivision
    if msh.porder>1         
        [pdg,udg] = scalarrefine(msh,pdg,udg,nref);        
    end                
    pm = reshape(mean(pdg,1),[size(udg,2) 3]); 
   
   % compute the electric amplitude along the line  xi + t*di
    for j = 1:N              
       d = (p(j,1)-pm(:,1)).^2+(p(j,2)-pm(:,2)).^2+(p(j,3)-pm(:,3)).^2;
       d = sqrt(d);
       [~,ks] = sort(d);          
       for k = 1:length(ks)               
           v1 = reshape(pdg(1,ks(k),:),[1 3]);
           v2 = reshape(pdg(2,ks(k),:),[1 3]);
           v3 = reshape(pdg(3,ks(k),:),[1 3]);
           v4 = reshape(pdg(4,ks(k),:),[1 3]);
           in = PointInTetrahedron(v1, v2, v3, v4, p(j,:));
           if in==1
               a = [v1 1; v2 1; v3 1; v4 1]\udg(:,ks(k));
               Ex(j)  = dot(a,[p(j,:) 1]);

               break;
           end
        end
    end
   
end


load('swarmair.mat','ALPHA','ETA','E_1atm');
ind = find(ALPHA>=ETA);
E1ATM = E_1atm(ind)/1e3;
ALPHA = ALPHA(ind);
ETA = ETA(ind);

Exmax = max(Ex);
E1min = min(E1ATM);
if Exmax>E1min
    % find the point at which Ex = E1min
    xbar = interp1(Ex,x,E1min);
    IND_LIM = (isnan(xbar));  % KEEP TRACK IF FIX WAS MADE, THAT MEANS Qcorona > evaluated value
    xbar(isnan(xbar))=x(end); % FIX IN CASE THE CONDITION IS NOT REACHED

    y = linspace(0,xbar,401);
    Ey = interp1(x,Ex,y);
    ALPHAy = interp1(E1ATM,ALPHA,Ey);
    ETAy = interp1(E1ATM,ETA,Ey);
    
    % AVOID NAN IN INTEGRATION
    ALPHA_eff = ALPHAy-ETAy;
    y = y(~isnan(ALPHA_eff));
    ALPHA_eff = ALPHA_eff(~isnan(ALPHA_eff));
    
    Q = 0;
    if length(y)>1
    Q = Rf*trapz(y,ALPHA_eff);
    end
end

if Exmax<=E1min
    Q = 0;
    IND_LIM = 0;
    y = [];
    Ey= [];
    ALPHAy = [];
    ETAy = [];
end



    
function sameside = IsSameSide(v1, v2, v3, v4, p)

normal = cross(v2 - v1, v3 - v1);
dotV4 = dot(normal, v4 - v1);
dotP = dot(normal, p - v1);

if abs(dotP)<=1e-10
    sameside = 1;
elseif sign(dotP) == sign(dotV4)
    sameside = 1;
else
    sameside = 0;
end

function in = PointInTetrahedron(v1, v2, v3, v4, p)

in = IsSameSide(v1, v2, v3, v4, p) & ...
     IsSameSide(v2, v3, v4, v1, p) & ...
     IsSameSide(v3, v4, v1, v2, p) & ...
     IsSameSide(v4, v1, v2, v3, p);               


function [pref,uref] = scalarrefine(mesh,p,u,nref)

[npl, nt, nd] = size(p);
porder=mesh.porder;
plocal=mesh.plocal;
tlocal=mesh.tlocal;

if isempty(nref), nref=ceil(log2(max(porder,1))); end
if size(tlocal,2)==4  
    A0=koornwinder(plocal(:,1:nd),porder);
    [plocal,tlocal]=uniref3d(plocal,tlocal,nref);    
    A=koornwinder(plocal(:,1:nd),porder)/A0;
else
    A0=tensorproduct(plocal(:,1:nd),porder);
    m = porder*(nref+1)+1;     
    [plocal,tlocal]=cubemesh(m,m,m,1);
    A=tensorproduct(plocal(:,1:nd),porder)/A0;  
end

npln=size(plocal,1);
t = kron(ones(nt,1),tlocal)+kron(npln*(0:nt-1)',0*tlocal+1);
ne = size(t,1);
np = npln*nt;

uref = reshape(A*u,[np 1]);
uref = reshape(uref(t'), [size(t,2) ne]);

pref = reshape(A*reshape(p,npl,nt*nd),[np,nd]);
pref = reshape(pref(t',:),[size(t,2) ne nd]);







