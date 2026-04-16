% Huaiyu Wei
% Prepare CCMP monthly zonal wind input on the mascon grid.

clearvars;
close all;
clc;

projectRoot = 'D:\OneDrive - University of California\MATLAB Codes\MOC\CMIP6_2026';
toolboxRoot = 'D:\OneDrive - University of California\MATLAB toolboxs';
ccmpDataDir = 'E:\Data_CCMPWind\RawData';
referenceMasconFile = 'E:\Data_CMIP6\ACCESS_historical\Mascon_V5_OBP_r10.mat';
accessReferenceFile = 'E:\Data_CMIP6\ACCESS_historical\uas_Amon_ACCESS-ESM1-5_historical_r36i1p1f1_gn_185001-201412.nc';
outputDir = 'E:\Data_CCMPWind';
outputFile = fullfile(outputDir, 'CCMPWind_mascon_200204_202504_V5.mat');

addpath(genpath(toolboxRoot));
addpath(genpath(projectRoot));

if ~isfolder(projectRoot)
    error('Project folder not found: %s', projectRoot);
end
if ~isfolder(toolboxRoot)
    error('Toolbox folder not found: %s', toolboxRoot);
end
if ~isfolder(ccmpDataDir)
    error('CCMP data folder not found: %s', ccmpDataDir);
end
if ~isfile(referenceMasconFile)
    error('Reference mascon file not found: %s', referenceMasconFile);
end
if ~isfolder(outputDir)
    error('CCMP output folder not found: %s', outputDir);
end


%% load monthly CCMP wind speed data
% https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10MMONTHLY_L4_V3.1#
cd(ccmpDataDir)
files = dir('CCMP_Wind_Analysis_*_monthly_mean_V03.1_L4.nc');
file_names = {files.name};
Nsamps = length(file_names);

if isempty(file_names)
    error('No CCMP monthly files were found in %s', ccmpDataDir);
end

u10 = nan(1440,720,Nsamps);
v10 = nan(1440,720,Nsamps);
time_wind = nan(Nsamps,1);
for ii = 1:Nsamps
 file = file_names{ii};
% ncdisp(file);
 u10(:,:,ii) = ncread(file,'u');
 v10(:,:,ii) = ncread(file,'v');
 time_wind(ii)=ncread(file,'time');
end

latitude= ncread(file,'latitude');
longitude= ncread(file,'longitude');

if any(diff(time_wind)<0)
    error('need to order data according to time')
end

time_wind = datetime(1987,1,1,0,0,0) + hours(time_wind);


%% convert 0~360 longitude to -180~180 longitude


longitude = cat(1,longitude(721:end),longitude(1:720));
longitude(longitude>180) = longitude(longitude>180)-360;
[latitude,longitude] = meshgrid(latitude,longitude);



u10= cat(1,u10(721:end,:,:),u10(1:720,:,:));


figure
pcolor(longitude,latitude,squeeze(mean(u10,3)))
shading flat
colorbar
clim([-12 12])
cmocean('red',41,'pivot',0)
title('Mean zonal wind')


%% load the mascon grid
load(referenceMasconFile)
Nmascon = length(lon_mascon_center);
clear Input_vars_mascon Input_time 


%% interpolate CCMP UAS  to mascons
u10_mascon = averageFieldToMascons(u10, longitude, latitude, ...
    lon_mascon_bound1, lon_mascon_bound2, lat_mascon_bound1, ...
    lat_mascon_bound2, flag_across_180);

figure
scatter(lon_mascon_center, lat_mascon_center,20,std(u10_mascon,1,2),'filled')
colorbar
title('STD')

figure
scatter(lon_mascon_center, lat_mascon_center,20,mean(u10_mascon,2),'filled')
clim([-12 12])
cmocean('red',41,'pivot',0)
colorbar
title('Mean zonal wind')

time_wind = time_wind(4:end);
u10_mascon = u10_mascon(:,4:end);
%%% Save result
cd(outputDir)
save(outputFile, ...
    'lon_mascon_center','lat_mascon_center','u10_mascon','time_wind','Basin_id','-v7.3');


