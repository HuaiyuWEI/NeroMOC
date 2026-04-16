% Huaiyu Wei
% Prepare GRACE/JPL mascon ocean-bottom-pressure input for the neural network.
% Data source:
% https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4

clearvars;
close all;
clc;

projectRoot = 'D:\OneDrive - University of California\MATLAB Codes\MOC\CMIP6_2026';
toolboxRoot = 'D:\OneDrive - University of California\MATLAB toolboxs';
masconDataRoot = 'D:\OneDrive - University of California\PythonWork\mascon';
outputFile = fullfile(masconDataRoot, 'GRACE_OBP_200204_202505_V5.mat');

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
if ~isfolder(masconDataRoot)
    error('Mascon data folder not found: %s', masconDataRoot);
end

%% load the mascon grid
grdfile = fullfile(projectRoot, 'BasinMask', 'Mascon_AtlSO.mat');
if ~isfile(grdfile)
    error('Mascon grid file not found: %s', grdfile);
end
load(grdfile)


%% load grace OBP data
% Grace satellite data
grdfile = fullfile(masconDataRoot, 'GRCTellus.JPL.200204_202505.GLO.RL06.3M.MSCNv04CRI.nc');
if ~isfile(grdfile)
    error('GRACE mascon file not found: %s', grdfile);
end
nc=netcdf(grdfile);
ncdisp(grdfile)

Grace_time= nc{'time'}(:);
Grace_time_bound= nc{'time_bounds'}(:);
lwe_thickness = nc{'lwe_thickness'}(:)/100; % from cm to m
scale_factor= nc{'scale_factor'}(:); 
GAD = nc{'GAD'}(:)/100; % from cm to m
uncertainty= nc{'uncertainty'}(:)/100.*9.806*1000;% from cm to m to pa

land_mask = nc{'land_mask'}(:);
land_mask = cat(2,land_mask(:,361:end),land_mask(:,1:360));
ocean_mask = -(land_mask-1);
ocean_mask(ocean_mask==0)=nan;


%%% Put Atlantic in the center (convert 0~360 longitude to -180~180 longitude)
lwe_thickness = cat(3,lwe_thickness(:,:,361:end),lwe_thickness(:,:,1:360));
scale_factor = cat(2,scale_factor(:,361:end),scale_factor(:,1:360));
GAD = cat(3,GAD(:,:,361:end),GAD(:,:,1:360));
uncertainty = cat(3,uncertainty(:,:,361:end),uncertainty(:,:,1:360));

figure
pcolor( ocean_mask.*squeeze(std( lwe_thickness,1)))
title('Liquid_Water_Equivalent_Thickness')
shading flat
colorbar
clim([0 0.1])

figure
pcolor( ocean_mask.*squeeze(std( GAD,1)))
shading flat
colorbar
clim([0 0.1])
title('GAD - Dealias')



%%%% lwe_thickness already have GAD added back - no need to add GAD again.
obp_grace = lwe_thickness.*9.806*1000;
GAD_grace = GAD.*9.806*1000;



Nmonth = size(lwe_thickness, 1);
obp_grace(:,or(lat_mascon<-75,lat_mascon>64.5)) = nan;
GAD_grace(:,or(lat_mascon<-75,lat_mascon>64.5)) = nan;

%% Mask GRACE fields over each mascon
obp_grace_mascon= zeros(Nmonth,Nmascon);
GAD_grace_mascon= zeros(Nmonth,Nmascon);
uncertainty_mascon= zeros(Nmonth,Nmascon);
for i = 1:Nmascon
    ind =  (mascon_ID == mascon_ID_uniq(i));
    obp_grace_mascon(:,i) = mean(ocean_mask(ind)'.*obp_grace(:,ind),[2 3],'omitnan');
    GAD_grace_mascon(:,i) = mean(ocean_mask(ind)'.*GAD_grace(:,ind),[2 3],'omitnan');
    uncertainty_mascon(:,i)= mean(ocean_mask(ind)'.*uncertainty(:,ind),[2 3],'omitnan');

end



%% interpolate GRACE data to stickly monthly frequency

% 1. Convert GRACE time (days since 2002-01-01) to datetime
t0 = datetime(2002, 1, 1);
time_grace = t0 + days(Grace_time);  % Grace_time is your original time vector


% 2. Create the strictly monthly time vector
time_monthly = [time_grace(1) datetime(2002, 5, 16):calmonths(1):datetime(2025, 5, 16)];
Nmonthly = numel(time_monthly);

% 3. Interpolate to monthly time
obp_grace_monthly = nan(Nmonthly, Nmascon);
GAD_grace_monthly = nan(Nmonthly, Nmascon);
for ii = 1:Nmascon
    obp_grace_monthly(:,ii) = interp1(time_grace, obp_grace_mascon(:,ii), time_monthly, 'linear');
    GAD_grace_monthly(:,ii) = interp1(time_grace, GAD_grace_mascon(:,ii), time_monthly, 'linear');
end


figure
pcolor(time_monthly,[1:Nmascon],obp_grace_monthly')
shading flat
colorbar
clim([-1000 1000])
cmocean('red',41)
xlabel('Time ')
ylabel('Location')

Basin_id_nan = Basin_id;
Basin_id_nan(Basin_id_nan==0) = nan;
figure
scatter(lon_mascon_center, lat_mascon_center, 15, std(obp_grace_monthly,1), 'filled');
xlabel('Longitude')
ylabel('Latitude')
title('STD of ocean bottom pressure')
colorbar;
cmocean('sha',20)
clim([0 500])
% xlim([-100 40])


figure
scatter(lon_mascon_center, lat_mascon_center, 15, std(GAD_grace_monthly,1), 'filled');
xlabel('Longitude')
ylabel('Latitude')
colorbar;
cmocean('sha',20)
title('STD of GAD from the dealiasing model')
clim([0 500])




%%% Save grace data
save(outputFile, ...
    'lon_mascon','lat_mascon','Basin_id','time_grace','time_monthly', ...
    'obp_grace_monthly', 'GAD_grace_monthly','uncertainty_mascon', ...
    'lon_mascon_center','lat_mascon_center', ...
    'lon_mascon_bound1','lon_mascon_bound2','lat_mascon_bound1','lat_mascon_bound2','-v7.3');

















