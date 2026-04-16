% Huaiyu Wei
% Prepare basin masks on the CMIP6 "gr" grid for later MOC processing.
% 2024-03-14 original version
% 2024-03-17 use sea area fraction == 100%% instead of depth == 0
% 2025-08-22 use sea area fraction > 50%% instead of == 100%%

clearvars;
close all;
clc;

projectRoot = 'D:\OneDrive - University of California\MATLAB Codes\MOC\CMIP6_2026';
toolboxRoot = 'D:\OneDrive - University of California\MATLAB toolboxs';
legacyCmipRoot = 'D:\OneDrive - University of California\MATLAB Codes\CMIP6';

addpath(genpath(projectRoot));
addpath(genpath(fullfile(toolboxRoot, 'netcdf')));
addpath(genpath(fullfile(toolboxRoot, 'cmocean')));
addpath(genpath(fullfile(toolboxRoot, 'cbrewer')));
addpath(genpath(fullfile(toolboxRoot, 'Global')));
addpath(genpath(legacyCmipRoot));

if ~isfolder(projectRoot)
    error('Project folder not found: %s', projectRoot);
end
if ~isfolder(toolboxRoot)
    error('Toolbox folder not found: %s', toolboxRoot);
end

%% User settings
CMIPname = 'CM4_PIc';
if strcmp(CMIPname,'CM4_PIc')
    DataPath = 'E:\Data_CMIP6\GFDL_PIcontrol\CM4'; 
    OutputFolder = fullfile(projectRoot, 'BasinMask');
end

if ~isfolder(DataPath)
    error('Input data folder not found: %s', DataPath);
end
if ~isfolder(OutputFolder)
    error('Output folder not found: %s', OutputFolder);
end



%% load grid and water depth
cd(DataPath)
DepthFile_gr = dir(fullfile(DataPath,['*deptho*gr.nc'])).name;
nc=netcdf(DepthFile_gr);
ncdisp(DepthFile_gr);

lat_gr=nc{'lat'}(:);  
lon_gr=nc{'lon'}(:);  
if(lon_gr(181)<180); error; end
lon_gr = [lon_gr(181:360)-360; lon_gr(1:180)];


[Y,X]=meshgrid(lat_gr,lon_gr);
B=nc{'deptho'}(:)';
fullvalue = ncreadatt(DepthFile_gr,'deptho','_FillValue');
B(B==fullvalue)=nan;
B = [B(181:360,:); B(1:180,:)];


handle = figure;
pcolor(X,Y,-B)
shading flat
cmp=cbrewer('div','RdYlBu',20);
colormap(cmp(end:-1:1,:))
clim([-6000 0])
ylabel('Latitude $^\circ$','Interpreter','latex')
xlabel('Longitude $^\circ$','Interpreter','latex')
title('GFDL-CM4 depth (m)','Interpreter','latex')




%% Double check land points using sea area fraction data

File_gr = dir(fullfile(DataPath,['*sftof*gr.nc'])).name;
nc=netcdf(File_gr);
ncdisp(File_gr);
C=nc{'sftof'}(:)';
fullvalue = ncreadatt(File_gr,'sftof','_FillValue');
C(C==fullvalue)=nan;
C = [C(181:360,:); C(1:180,:)];


handle = figure;
pcolor(X,Y,C)
shading flat
colorbar
cmp=cbrewer('div','RdYlBu',20);
colormap(cmp(:,:))
clim([0 100])
ylabel('Latitude $^\circ$','Interpreter','latex')
xlabel('Longitude $^\circ$','Interpreter','latex')
title('GFDL-CM4 sea area fraction ($\%$)','Interpreter','latex')


%% Double check land points using sst data

% SSTFile_gr = 'tos_Omon_GFDL-CM4_piControl_r1i1p1f1_gr_015101-017012.nc';
% nc=netcdf(SSTFile_gr);
% ncdisp(SSTFile_gr);
% SST=nc{'tos'}(:);
% fullvalue = ncreadatt(SSTFile_gr,'tos','_FillValue');
% SST(SST==fullvalue)=nan;
% SST = permute(SST,[3 2 1]);
% SST = squeeze([SST(181:360,:,1); SST(1:180,:,1)]);
% 
% handle = figure;
% pcolor(X,Y,SST)
% shading flat
% colorbar
% cmocean('bal',100)
% clim([0 30])
% ylabel('Latitude $^\circ$','Interpreter','latex')
% xlabel('Longitude $^\circ$','Interpreter','latex')
% title('GFDL-SST depth ($^\circ$C)','Interpreter','latex')

%% Remove cells with more than half land
% B2=B;B2(B2==0)=nan; % 
% B2=C;B2(B2~=100)=nan; % 
B2=C;B2(B2<=50)=nan; % 

handle = figure;
pcolor(X,Y,-B2)
shading flat
colorbar
cmocean('-the',100)
clim([-6000 0])
ylabel('Latitude $^\circ$','Interpreter','latex')
xlabel('Longitude $^\circ$','Interpreter','latex')
title('GFDL-CM4 depth (m)','Interpreter','latex')



%%  not used
%%% ! consider shallow regions as land;
% this is motivated by the need to separate Pacific and Atlantic at ~80W 9N
% 
% B2(B2<300)=nan; % 
% handle = figure;
% pcolor(X,Y,-B2)
% shading flat
% colorbar
% cmocean('-the',100)
% clim([-6000 0])
% ylabel('Latitude $^\circ$','Interpreter','latex')
% xlabel('Longitude $^\circ$','Interpreter','latex')
% title('GFDL-CM4 depth (m)','Interpreter','latex')



%% Close off Med Sea, which is open in CM4 gr-grid by 2 points
% [~,nx] = min(abs( (X(:)-(-5.5)).^2+(Y(:)-36).^2));
% [nx,ny] = ind2sub(size(X),nx);
% B2(nx,ny:ny+1) = nan;
% 
% handle = figure;
% pcolor(X,Y,-B2)
% shading flat
% colorbar
% cmocean('-the',100)
% clim([-6000 0])
% ylabel('Latitude $^\circ$','Interpreter','latex')
% xlabel('Longitude $^\circ$','Interpreter','latex')
% title('GFDL-CM4 depth (m)','Interpreter','latex')

%% Close off the west side of the Labrador sea
[~,nx] = min(abs( (X(:)-(-77)).^2+(Y(:)-64).^2));
[nx,ny] = ind2sub(size(X),nx);
B2(nx,ny:ny+1) = nan;

handle = figure;
pcolor(X,Y,-B2)
shading flat
colorbar
cmocean('-the',100)
clim([-6000 0])
ylabel('Latitude $^\circ$','Interpreter','latex')
xlabel('Longitude $^\circ$','Interpreter','latex')
title('GFDL-CM4 depth (m)','Interpreter','latex')


%% Close off the east side of the North sea
[~,nx] = min(abs( (X(:)-(10)).^2+(Y(:)-58).^2));
[nx,ny] = ind2sub(size(X),nx);
B2(nx,ny:ny+1) = nan;

handle = figure;
pcolor(X,Y,-B2)
shading flat
colorbar
cmocean('-the',100)
clim([-6000 0])
ylabel('Latitude $^\circ$','Interpreter','latex')
xlabel('Longitude $^\circ$','Interpreter','latex')
title('GFDL-CM4 depth (m)','Interpreter','latex')

%% Close the gap between North America and South America
lat0 = 8.5; %[deg]E
for lon0=-82.5:0.01:-75.5
    [~,nx] = min(abs( (X(:)-lon0).^2+(Y(:)-lat0).^2));
    [nx,ny] = ind2sub(size(X),nx);
    B2(nx,ny) = nan;
end

[~,nx] = min(abs( (X(:)-(-83.5)).^2+(Y(:)-9.5).^2));
[nx,ny] = ind2sub(size(X),nx);
B2(nx,ny:ny) = nan;



handle = figure;
pcolor(X,Y,-B2)
shading flat
colorbar
cmocean('-the',100)
clim([-6000 0])
ylabel('Latitude $^\circ$','Interpreter','latex')
xlabel('Longitude $^\circ$','Interpreter','latex')
title('GFDL-CM4 depth (m)','Interpreter','latex')



%% Close off Southern Ocean east of the South Africa tip, at 21.5E
lon0 = 21.5; %[deg]E
for lat0=-33:-0.1:-71
    [~,nx] = min(abs( (X(:)-lon0).^2+(Y(:)-lat0).^2));
    [nx,ny] = ind2sub(size(X),nx);
    B2(nx,ny) = nan;
end

handle = figure;
pcolor(X,Y,-B2)
shading flat
colorbar
cmocean('-the',100)
clim([-6000 0])
ylabel('Latitude $^\circ$','Interpreter','latex')
xlabel('Longitude $^\circ$','Interpreter','latex')
title('GFDL-CM4 depth (m)','Interpreter','latex')


%% Close off Southern Ocean west of Drake Passage, at -66.5E
lon0 = -66.5; %[deg]E
for lat0=-55:-0.1:-67
    [~,nx] = min(abs( (X(:)-lon0).^2+(Y(:)-lat0).^2));
    [nx,ny] = ind2sub(size(X),nx);
    B2(nx,ny) = nan;
end

handle = figure;
pcolor(X,Y,-B2)
shading flat
colorbar
cmocean('-the',100)
clim([-6000 0])
ylabel('Latitude $^\circ$','Interpreter','latex')
xlabel('Longitude $^\circ$','Interpreter','latex')
title('GFDL-CM4 depth (m)','Interpreter','latex')


%% Close off Arctic Ocean north of 67.5N
lat0 = 67.5; %[deg]E
for lon0=-190:0.01:190
    [~,nx] = min(abs( (X(:)-lon0).^2+(Y(:)-lat0).^2));
    [nx,ny] = ind2sub(size(X),nx);
    B2(nx,ny) = nan;
end


handle = figure;
pcolor(X,Y,-B2)
shading flat
colorbar
cmocean('-the',100)
clim([-6000 0])
ylabel('Latitude $^\circ$','Interpreter','latex')
xlabel('Longitude $^\circ$','Interpreter','latex')
title('GFDL-CM4 depth (m)','Interpreter','latex')

%% Land+Basin-boundaries Mask
Mask = isnan(B2)*2; %So==0 in ocean points, and ==2 on land points.
%% Create Atlantic Mask

AtlSeedLon = -20; AtlSeedLat = 0; %[deg] E/N
[~,AtlSeedX] = min(abs( (X(:)-AtlSeedLon).^2+(Y(:)-AtlSeedLat).^2));
[AtlSeedX,AtlSeedY] = ind2sub(size(X),AtlSeedX);
MaskAtl = CalcFloodFill_Huaiyu(Mask,AtlSeedX,AtlSeedY);
% fh = figure; contourf(X,Y,MaskAtl); colormap jet; colorbar;xlim([-180,180]);
fh = figure; pcolor(X,Y,MaskAtl); shading flat; colormap haxby; colorbar;xlim([-180,180]);
print(fh,[OutputFolder,'MaskAtl.png'],'-dpng','-r0'); 

%% Create Indo-Pacific Mask
%Western hemisphere flood-fill:
PacSeedLon = -150; PacSeedLat = 0; %[deg] E/N
[~,PacSeedX] = min(abs( (X(:)-PacSeedLon).^2+(Y(:)-PacSeedLat).^2));
[PacSeedX,PacSeedY] = ind2sub(size(X),PacSeedX);
MaskPac = CalcFloodFill_Huaiyu(Mask,PacSeedX,PacSeedY);
%Eastern hemisphere flood-fill:
PacSeedLon = 150; PacSeedLat = 0; %[deg] E/N
[~,PacSeedX] = min(abs( (X(:)-PacSeedLon).^2+(Y(:)-PacSeedLat).^2));
[PacSeedX,PacSeedY] = ind2sub(size(X),PacSeedX);
MaskPac = CalcFloodFill_Huaiyu(MaskPac,PacSeedX,PacSeedY);
% fh = figure; contourf(X,Y,MaskPac); colormap haxby; colorbar;xlim([-180,180]);
fh = figure; pcolor(X,Y,MaskPac); shading flat; colormap haxby; colorbar;xlim([-180,180]);
print(fh,[OutputFolder,'MaskPac.png'],'-dpng','-r0'); 




MaskAll = MaskPac+MaskAtl;
MaskAll(MaskAll~=1)=nan;




SOind = find(lat_gr<-35,1,'last');

for  lat_temp = 1:SOind+1
MaskAll(~isnan(MaskAll(:,lat_temp)),lat_temp) = 3;
end


lon0 = 21.5; %[deg]E
for lat0=-35:-0.1:-69
    [~,nx] = min(abs( (X(:)-lon0).^2+(Y(:)-lat0).^2));
    [nx,ny] = ind2sub(size(X),nx);
    MaskAll(nx,ny) = 3;
end
% 
lon0 = -66.5; %[deg]E
for lat0=-55:-0.1:-65
    [~,nx] = min(abs( (X(:)-lon0).^2+(Y(:)-lat0).^2));
    [nx,ny] = ind2sub(size(X),nx);
    MaskAll(nx,ny) = 3;
end
% 

for lat_temp = SOind+2 : length(lat_gr)
    MaskAll(MaskPac(:,lat_temp)==1,lat_temp) = 2;
end
fh = figure; ; shading flat; colormap jet; colorbar;xlim([-180,180]);


fontsize = 18; LW =2;
handle = figure; set(handle,'Position',[100 100 1000 600]); 
pcolor(X,Y,MaskAll)
shading flat
colorbar('Ticks',[1 2 3])
cmp=cbrewer('qual','Set2',3);
colormap(cmp(:,:))
set(gca,'FontSize',fontsize)
set(gca,'layer','top','linewidth',LW-1)
ylabel('Latitude $(^\circ)$','FontSize',fontsize,'Interpreter','latex')
xlabel('Longitude $(^\circ)$','FontSize',fontsize,'Interpreter','latex')
title("Basin Masks of `gr' grid",'FontSize',fontsize,'Interpreter','latex')
ylim([-80 80])
exportgraphics(gcf,fullfile(OutputFolder,'Basin Masks_gr.png'),'resolution',200)


MaskAtlSO = MaskAll;
MaskAtlSO(MaskAtlSO==2) = nan;
MaskAtlSO(MaskAtlSO==3) = 1;



fontsize = 18; LW =2;
handle = figure; set(handle,'Position',[100 100 1000 600]); 
pcolor(X,Y,MaskAtlSO)
shading flat
colorbar('Ticks',[1 2 3])
cmp=cbrewer('qual','Set2',3);
colormap(cmp(:,:))
set(gca,'FontSize',fontsize)
set(gca,'layer','top','linewidth',LW-1)
ylabel('Latitude $(^\circ)$','FontSize',fontsize,'Interpreter','latex')
xlabel('Longitude $(^\circ)$','FontSize',fontsize,'Interpreter','latex')
title("MaskAtlSO",'FontSize',fontsize,'Interpreter','latex')
ylim([-80 80])

MaskOcean = MaskAll;
MaskOcean(MaskOcean==2) = 1;
MaskOcean(MaskOcean==3) = 1;

fontsize = 18; LW =2;
handle = figure; set(handle,'Position',[100 100 1000 600]); 
pcolor(X,Y,MaskOcean)
shading flat
colorbar('Ticks',[1 2 3])
cmp=cbrewer('qual','Set2',3);
colormap(cmp(:,:))
set(gca,'FontSize',fontsize)
set(gca,'layer','top','linewidth',LW-1)
ylabel('Latitude $(^\circ)$','FontSize',fontsize,'Interpreter','latex')
xlabel('Longitude $(^\circ)$','FontSize',fontsize,'Interpreter','latex')
title("MaskOcean",'FontSize',fontsize,'Interpreter','latex')
ylim([-80 80])

%% Save masks
save(fullfile(OutputFolder,'BasinMasks_gr_V2026.mat'), ...
    'X','Y','MaskAtl','MaskPac','MaskAtlSO','MaskOcean','MaskAll');

cd(OutputFolder)
