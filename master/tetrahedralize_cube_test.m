d = [-1 1];
[x,y,z] = meshgrid(d);
X = [x(:) y(:) z(:)];

T = [1 2 3 6;
     2 4 3 6;
     3 6 4 8;
     1 3 5 6;
     3 7 5 6;
     3 8 7 6];
 
fig=figure('visible','off');
line(X(1:2,1),X(1:2,2),X(1:2,3),'marker','.','markersize',20,'markeredgecolor',[1 0.5 0],'linestyle','-', 'linewidth',2,'color','blue')
for i=1:6
    subplot(1,6,i)
    tetramesh(T(i,:),X,'FaceColor',[1 0.5 0],'EdgeColor',[0 0 1],'FaceAlpha',0.3);
   
end

saveas(fig,'tets','png')