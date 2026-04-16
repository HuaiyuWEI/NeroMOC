% Huaiyu Wei
% Define the JPL GRACE mascon grid used for neural-network input/output data.
% Data source:
% https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/

clearvars;
close all;
clc;

projectRoot = 'D:\OneDrive - University of California\MATLAB Codes\MOC\CMIP6_2026';
toolboxRoot = 'D:\OneDrive - University of California\MATLAB toolboxs';

addpath(genpath(toolboxRoot));
addpath(genpath(projectRoot));
fontsize = 18;
LW = 2;

if ~isfolder(projectRoot)
    error('Project folder not found: %s', projectRoot);
end
if ~isfolder(toolboxRoot)
    error('Toolbox folder not found: %s', toolboxRoot);
end

%% load the mascon grid  
grdfile = 'D:\OneDrive - University of California\PythonWork\mascon\GRCTellus.JPL.200204_202505.GLO.RL06.3M.MSCNv04CRI.nc';
if ~isfile(grdfile)
    error('Mascon grid file not found: %s', grdfile);
end
nc=netcdf(grdfile);
ncdisp(grdfile)
lon_mascon = nc{'lon'}(:);
lat_mascon = nc{'lat'}(:);
lon_mascon = [lon_mascon(361:end); lon_mascon(1:360)];
lon_mascon(lon_mascon>180) = lon_mascon(lon_mascon>180)-360;
mascon_ID = nc{'mascon_ID'}(:);
land_mask = nc{'land_mask'}(:);
land_mask = cat(2,land_mask(:,361:end),land_mask(:,1:360));
mascon_ID = cat(2,mascon_ID(:,361:end),mascon_ID(:,1:360));

[lon_mascon,lat_mascon] = meshgrid(lon_mascon,lat_mascon);
ocean_mask = -(land_mask-1);

pcolor(lon_mascon,lat_mascon,ocean_mask);shading flat;colorbar
title('ocean mask from GRACE')

disp(max(mascon_ID(:)))


%% make an example plot for the mascon grid
mascon_ID_plot = mascon_ID;
lon_plot_1 = -176;
lon_plot_2 = -170;
lat_plot_1 =-35;
lat_plot_2 =-30;
mascon_ID_plot(lon_mascon<lon_plot_1-1)=nan;
mascon_ID_plot(lon_mascon>lon_plot_2+1)=nan;
mascon_ID_plot(lat_mascon<lat_plot_1-1)=nan;
mascon_ID_plot(lat_mascon>lat_plot_2+1)=nan;
unique_vals = unique(mascon_ID_plot(~isnan(mascon_ID_plot)));
unique_vals = sort(unique_vals);
for i = 1:length(unique_vals)
    mascon_ID_plot(mascon_ID_plot==unique_vals(i)) = i-0.5;
end
figure
pcolor(lon_mascon,lat_mascon,mascon_ID_plot)
cmp=cbrewer('qual','Set1',length(unique_vals));
clim([0 length(unique_vals)])
colormap(cmp)
colorbar
xlabel('Longitude')
ylabel('Latitude')
yticks([-35:1:-30])
xticks([-176:2:-170])
axis equal
axis([lon_plot_1 lon_plot_2 lat_plot_1 lat_plot_2])

%% remove mascons in polar region
mascon_ID(or(lat_mascon<-75,lat_mascon>64.5)) = nan;

mascon_ID_uniq = unique(mascon_ID(~isnan(mascon_ID)));
for i = 1:length(mascon_ID_uniq)
    ind =  (mascon_ID == mascon_ID_uniq(i));
if any(land_mask(ind) ~= 0) % remove mascons that contain any land cells
        % if(mean(land_mask(ind)) ==1) % remove mascon 100% on land
        % if(mean(land_mask(ind)) >=0.5) % remove mascon more than half land
        mascon_ID_uniq(i) = nan;
    end
end
% label all mascon
mascon_ID_uniq = mascon_ID_uniq(~isnan(mascon_ID_uniq));
Nmascon = length(mascon_ID_uniq);


%% find the lat and lon for each labeled mascon
lon_mascon_bound1 = zeros(Nmascon,1);
lon_mascon_bound2 = zeros(Nmascon,1);
lat_mascon_bound1 = zeros(Nmascon,1);
lat_mascon_bound2 = zeros(Nmascon,1);
flag_across_180 = zeros(Nmascon,1);
for i = 1:Nmascon
    ind =  (mascon_ID == mascon_ID_uniq(i));
    if(any(lon_mascon(ind) > 0) && any(lon_mascon(ind) < 0))
        temp = lon_mascon(ind);
        lon_mascon_bound1(i) = min(temp(temp>0)) - 0.25;
        lon_mascon_bound2(i) = max(temp(temp<0)) + 0.25;
        flag_across_180(i) = 1;
    else
        lon_mascon_bound1(i) = min(lon_mascon(ind)) - 0.25;
        lon_mascon_bound2(i) = max(lon_mascon(ind)) + 0.25;
    end
    lat_mascon_bound1(i) = min(lat_mascon(ind)) - 0.25;
    lat_mascon_bound2(i) = max(lat_mascon(ind)) + 0.25;
end


%% load pre-calculated basin mask, based on the CMIP6 "gr" grid
BasinMasksFN = fullfile(projectRoot, 'BasinMask', 'BasinMasks_gr_V2026.mat');
if ~isfile(BasinMasksFN)
    error('Basin mask file not found: %s', BasinMasksFN);
end
load(BasinMasksFN)

BasinMask = MaskAtlSO;

figure('Position',[50,50,1600,1000],'Color','white');
m_proj('Robinson','lon',[-180 180],'lat',[-80 80]);
m_pcolor(X,Y,BasinMask);
m_coast('patch',[.7 .7 .7],'edgecolor','none');
m_grid('tickdir','in','linewi',1,'gridcolor','none','FontSize',fontsize-1);
set(gca,'FontSize',fontsize-3,'linewidth',LW-1);
set(gca,'Layer','top','tickLabelinterpreter', 'latex')
title('Mask for the Atlantic and Southern Ocean')
box on
colorbar



%% Find Mascons in the Atlantic and Southern Ocean
lon_mascon_center = 0.5*(lon_mascon_bound1+lon_mascon_bound2);
lat_mascon_center = 0.5*(lat_mascon_bound1+lat_mascon_bound2);
lon_mascon_center(flag_across_180==1) = lon_mascon_center(flag_across_180==1)+180;
Basin_id = nan(Nmascon, 1);  % Preallocate output
for i = 1:Nmascon
    dist2 = (X - lon_mascon_center(i)).^2 + (Y - lat_mascon_center(i)).^2;
    % Find the index of the closest point
    [~, ind] = min(dist2(:));
    [row, col] = ind2sub(size(BasinMask), ind);
    % Assign the basin mask value
    Basin_id(i) = BasinMask(row, col);
end
% Basin_id( Basin_id==2)=nan;
% Basin_id( lat_mascon_center<-36.1)=0;


figure('Position',[50,50,1000,600],'Color','white');
m_proj('Robinson','lon',[-180 180],'lat',[-80 80]);
m_scatter(lon_mascon_center, lat_mascon_center, 15,  Basin_id, 'filled');
m_coast('patch',[.7 .7 .7],'edgecolor','none');
m_grid('tickdir','in','linewi',1,'gridcolor','none','FontSize',fontsize-1);
set(gca,'FontSize',fontsize-3,'linewidth',LW-1);
set(gca,'Layer','top','tickLabelinterpreter', 'latex')
box on
colorbar
title('Mascons in the Atlantic and Southern Ocean')


%% Save mascon metadata
outputFile = fullfile(projectRoot, 'BasinMask', 'Mascon_AtlSO.mat');
save(outputFile, ...
    'lon_mascon','lat_mascon','Basin_id','Nmascon', ...
    'lon_mascon_center','lat_mascon_center','mascon_ID','mascon_ID_uniq', 'flag_across_180', ...
    'lon_mascon_bound1','lon_mascon_bound2','lat_mascon_bound1','lat_mascon_bound2','-v7.3');
