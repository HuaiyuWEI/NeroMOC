% Huaiyu Wei
% Prepare DUACS sea-surface-height input on the mascon grid.

clearvars;
close all;
clc;

projectRoot = 'D:\OneDrive - University of California\MATLAB Codes\MOC\CMIP6_2026';
toolboxRoot = 'D:\OneDrive - University of California\MATLAB toolboxs';
duacsDataDir = 'E:\Data_SSH\DUACS\daily';
duacsMonthlyFile = fullfile(duacsDataDir, 'DUACS_200204_202504.mat');
referenceMasconFile = fullfile(projectRoot, 'BasinMask', 'Mascon_AtlSO.mat');
outputFile = fullfile(duacsDataDir, 'DUACS_mascon_200204_202504_V5.mat');

addpath(genpath(toolboxRoot));
addpath(genpath(projectRoot));

if ~isfolder(projectRoot)
    error('Project folder not found: %s', projectRoot);
end
if ~isfolder(toolboxRoot)
    error('Toolbox folder not found: %s', toolboxRoot);
end
if ~isfolder(duacsDataDir)
    error('DUACS data folder not found: %s', duacsDataDir);
end
if ~isfile(referenceMasconFile)
    error('Reference mascon file not found: %s', referenceMasconFile);
end


%% Calculates monthly Absolute dynamic topography from daily Copernicus altimetry data.
if ~isfile(duacsMonthlyFile)
cd(duacsDataDir)
filelist = dir(fullfile(duacsDataDir, '*.nc'));
if ~isempty(filelist)
    [~, order] = sort({filelist.name});
    filelist = filelist(order);
end
disp({filelist.name}')

if isempty(filelist)
    error('No DUACS NetCDF files were found in %s', duacsDataDir);
end

% Initialize structures to accumulate
adt_monthly_map = containers.Map('KeyType','double','ValueType','any');

for f = 1:length(filelist)
    filename = filelist(f).name;
    disp(['Processing: ', filename])
    % ncdisp(filename)
    % Read time and ADT
    time = ncread(filename, 'time'); % days since 1950-01-01
    adt  = ncread(filename, 'adt');  % [lon, lat, time]
    fillValue = ncreadatt(filename, 'adt', '_FillValue');
    adt(adt == fillValue) = nan;

    % Convert to datetime
    time_dt = datetime(1950,1,1) + days(time);
    [yr, mo] = ymd(time_dt);
    month_id = yr * 100 + mo; % unique identifier for each month

    % Loop through each month in this file
    for m = unique(month_id(:))'
        idx = find(month_id == m);
        if isempty(idx), continue; end
        % Accumulate (if multiple files cover same month, average again)
        if isKey(adt_monthly_map, m)
            error('same month is found in multiple files')
        else
            adt_monthly_map(m) = struct('sum', sum(adt(:,:,idx),3,'omitnan'), ...
                                        'count', numel(idx));
        end
    end
end

% Final assembly
all_months = sort(cell2mat(keys(adt_monthly_map)));
sample_file = fullfile(duacsDataDir,filelist(1).name);
lat_DUACS = ncread(sample_file, 'latitude');
lon_DUACS = ncread(sample_file, 'longitude');
nx = length(lon_DUACS); ny = length(lat_DUACS);

adt_monthly = NaN(nx, ny, length(all_months));
for i = 1:length(all_months)
    tmp = adt_monthly_map(all_months(i));
    adt_monthly(:,:,i) = tmp.sum ./ tmp.count;
end
monthly_time = datetime(floor(all_months/100), mod(all_months,100), 1);



[lat_DUACS,lon_DUACS] = meshgrid(lat_DUACS,lon_DUACS);


% Save result
save(duacsMonthlyFile, ...
    'lat_DUACS','lon_DUACS','adt_monthly','monthly_time','-v7.3');

end

%% load monthly SSH data
if isfile(duacsMonthlyFile)
load(duacsMonthlyFile)
end

figure
pcolor(lon_DUACS,lat_DUACS,squeeze(std(adt_monthly,1,3)))
shading flat
title('std')
colorbar

figure
pcolor(lon_DUACS,lat_DUACS,squeeze(mean(adt_monthly,3)))
title('mean')
shading flat
cmocean('red',50,'pivot',0)
colorbar


figure
pcolor(lon_DUACS,lat_DUACS,squeeze((adt_monthly(:,:,88))))
title('snapshot')
shading flat
colorbar

figure
plot(squeeze(adt_monthly(500,500,:)))
title('time series at a certain location')


%% load the mascon grid
load(referenceMasconFile)
Nmascon = length(lon_mascon_center);
clear Input_vars_mascon Input_time 



%% interpolate DUACS SSH  to mascons
adt_mascon = averageFieldToMascons(adt_monthly, lon_DUACS, lat_DUACS, ...
    lon_mascon_bound1, lon_mascon_bound2, lat_mascon_bound1, ...
    lat_mascon_bound2, flag_across_180);


figure
scatter(lon_mascon_center, lat_mascon_center,20,std(adt_mascon,1,2),'filled')
title('std')
colorbar

figure
scatter(lon_mascon_center, lat_mascon_center,20,squeeze(adt_mascon(:,88)),'filled')
title('snapshot')
colorbar
%% compute mean adt within Jan 2004 to Dec 2009
% Extract year and month
time_year = year(monthly_time);
time_month = month(monthly_time);
% Find indices within Jan 2004 to Dec 2009
ind_time = find( time_year >= 2004 & time_year <= 2009 );
if isempty(ind_time)
    error('No DUACS monthly samples between Jan 2004 and Dec 2009 were found in %s', duacsMonthlyFile);
end
t_start = ind_time(1);
t_end   = ind_time(end);
t_count = t_end - t_start + 1;
if(t_count~=72)
error('check avg index')
end

adt_mascon_ref = mean(adt_mascon(:,t_start:t_end),2,'omitnan');
ssh_mascon = adt_mascon - adt_mascon_ref;


figure
scatter(lon_mascon_center, lat_mascon_center,20,adt_mascon_ref,'filled')
cmocean('red',50,'pivot',0)
title("2004-2009 mean")
colorbar


figure
scatter(lon_mascon_center, lat_mascon_center,20,ssh_mascon(:,88),'filled')
cmocean('red',50,'pivot',0)
colorbar



% At several Mascons near Antarctica, SSH data are unavailable in months 
% affected by ice cover. These missing values are filled using the time-mean 
% SSH at the same location, estimated from months with available observations.

ssh_mascon_NoNan = ssh_mascon;
for ii = 1:Nmascon
    ind_nan = isnan(ssh_mascon(ii,:));
    ssh_mascon_NoNan(ii,ind_nan) = mean(ssh_mascon(ii,:), 'omitnan');
end


%%% Save result
cd(duacsDataDir)
save(outputFile, ...
    'lon_mascon_center','lat_mascon_center','ssh_mascon_NoNan','ssh_mascon','adt_mascon_ref','monthly_time','Basin_id','-v7.3');
