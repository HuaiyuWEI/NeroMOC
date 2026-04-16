% Huaiyu Wei
% Interpolate meridional overturning circulation output from its native
% grid to the 1-degree CMIP6 "gr" grid.

clearvars;
close all;
clc;

projectRoot = 'D:\OneDrive - University of California\MATLAB Codes\MOC\CMIP6_2026';
toolboxRoot = 'D:\OneDrive - University of California\MATLAB toolboxs';

addpath(genpath(toolboxRoot));
addpath(genpath(projectRoot));

if ~isfolder(projectRoot)
    error('Project folder not found: %s', projectRoot);
end
if ~isfolder(toolboxRoot)
    error('Toolbox folder not found: %s', toolboxRoot);
end

%% User settings
DataPath_CM4 = 'E:\Data_CMIP6\GFDL_PIcontrol\CM4\';
rhoConst = 1035;
ensembleStart = 1;
ensembleEnd = 40;
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_historical\';
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP245\';
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP370\';
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP126\';
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP126\2100-2300\';
DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP585';

if ~isfolder(DataPath_CM4)
    error('CM4 grid folder not found: %s', DataPath_CM4);
end
if ensembleStart > ensembleEnd
    error('ensembleStart must be less than or equal to ensembleEnd.')
end
if ~isfolder(DataPath_ESM1d5)
    error('Input data folder not found: %s', DataPath_ESM1d5);
end

outputDir = DataPath_ESM1d5;


% load the gr grid from GFDL CM4
cd(DataPath_CM4)
depthFiles = dir(fullfile(DataPath_CM4, '*deptho*gr.nc'));
if isempty(depthFiles)
    error('No CM4 depth file matching *deptho*gr.nc was found in %s', DataPath_CM4);
end
if numel(depthFiles) > 1
    error('Multiple CM4 depth files matched *deptho*gr.nc in %s', DataPath_CM4);
end
DepthFile_gr = depthFiles(1).name;
ncdisp(DepthFile_gr)
nc=netcdf(DepthFile_gr);
lat_gr=nc{'lat'}(:)';
lon_gr=nc{'lon'}(:)';
Nlon_gr = length(lon_gr);
Nlat_gr = length(lat_gr);
[Lat_gr,Lon_gr] = meshgrid(lat_gr,lon_gr);
% Load the basin mask prepared on the "gr" grid.
BasinMasksFN = fullfile(projectRoot, 'BasinMask', 'BasinMasks_gr_V2026.mat');
if ~isfile(BasinMasksFN)
    error('Basin mask file not found: %s', BasinMasksFN);
end
load(BasinMasksFN)

%% interpolate ACCESS-ESM1.5 MOC strength

%%% load residual MOC
cd(DataPath_ESM1d5)
% https://gmd.copernicus.org/preprints/gmd-2016-77/gmd-2016-77.pdf
% Page 41. Details in MOC defination and calculation
% PsiName = 'msftyrho'
PsiName = 'msftmrho'
% for heat flux, the variable name would be hfbasin

files = dir(fullfile(DataPath_ESM1d5,['*' PsiName '*.nc']));
if ~isempty(files)
    [~, order] = sort({files.name});
    files = files(order);
end
file_names = {files.name};
% Extract number after "_r" (e.g., r2, r11, etc.)
r_numbers = zeros(size(file_names));
for i = 1:length(file_names)
    % Match '_r<num>i' using regular expression
    tokens = regexp(file_names{i}, '_r(\d+)i', 'tokens');
    if ~isempty(tokens)
        r_numbers(i) = str2double(tokens{1}{1});
    end
end

filelist = file_names;
disp(filelist')

if isempty(filelist)
    error('No %s files were found in %s', PsiName, DataPath_ESM1d5);
end



for r_ind = ensembleStart:ensembleEnd

    indall = find(r_numbers == r_ind);
    disp(['Number of files in realization r' num2str(r_ind) ': ' num2str(length(indall))])
    if isempty(indall)
        error('Missing file for realization r%d.', r_ind)
    end

    Psi_all = [];
    Psi_time_all  = [];

    for i = indall

        file_base = filelist{i};
        fprintf('Processing realization file: %s\n', file_base);

        % Extract realization number
        tokens = regexp(file_base, '_r(\d+)i\d+p\d+f\d+', 'tokens');
        realization = '000';  % default if no match
        if ~isempty(tokens)
            realization = tokens{1}{1};
        end

        Psi = [];

        file = fullfile(DataPath_ESM1d5, filelist{i});
        % ncdisp(file); finfo=ncinfo(file);
        nc=netcdf(file);
        Psi=nc{PsiName}(:);
        fullvalue = ncreadatt(file,PsiName,'_FillValue');
        Psi(Psi==fullvalue)=nan;
        Psi = permute(Psi,[4 3 2 1]);
        time_Psi=nc{'time'}(:); %days
        sector=nc{'sector'}(:);

        Psi_all = cat(4,Psi_all,Psi);
        Psi_time_all = cat(1,Psi_time_all,time_Psi);

    end

if(any(diff(Psi_time_all)<0))
    error('Check time')
end
Nsamps_ESM1d5 = length(Psi_time_all);


Psi_all = Psi_all./rhoConst;
time_Psi = Psi_time_all;


rho2_i=nc{'rho'}(:);  % Target Potential Density at interface; positive downward; referenced to 2000 db
lat_psi = nc{'lat'}(:);
Psi_AMOC = squeeze(Psi_all(:,:,1,:)); % atlantic_arctic_ocean
Psi_SOMOC = squeeze(Psi_all(:,:,3,:)); % global_ocean
clear Psi_all

% test1 = Psi_AMOC(:,:,199);
% test2 = Psi_SOMOC(:,:,199);

separationDensityIndex = min(40, size(Psi_AMOC, 2));
ind_AMOC_SOMOC_sep = find(~isnan(Psi_AMOC(:, separationDensityIndex, 1)), 1);
if isempty(ind_AMOC_SOMOC_sep) || ind_AMOC_SOMOC_sep == 1
    error('Could not determine the AMOC/SOMOC separation latitude from column %d.', separationDensityIndex);
end
Psi_ASMOC = cat( 1, Psi_SOMOC(1:ind_AMOC_SOMOC_sep-1,:,:), Psi_AMOC(ind_AMOC_SOMOC_sep:end,:,:));

disp(['Northmost grid for SOMOC is' num2str(lat_psi(ind_AMOC_SOMOC_sep-1))])
disp(['Southmost grid for AMOC is' num2str(lat_psi(ind_AMOC_SOMOC_sep))])


moc_mean = squeeze(mean(Psi_ASMOC/1e6,3));

% figure
% contourf(lat_psi,-rho2_i,squeeze(mean(Psi_ASMOC/1e6,3))',[-50:0.5:50],'Edgecolor','none')
% shading flat
% xlim([-75 65])
% ylim([-1037.2 -1035])
% clim([-20 20])
% cmocean('red',100)


%% full-depth MOC in the Atlantic and SO
[RHO_PSI,LAT_PSI ] = meshgrid(rho2_i,lat_psi);


rho2_interp = rho2_i(and(rho2_i>1035,rho2_i<1037.2));
if isempty(rho2_interp)
    error('No density levels were found between 1035 and 1037.2 kg m^-3.');
end
MOC_gr_ind = [16:155]; %from 75S to 65N
if MOC_gr_ind(end) > numel(lat_gr)
    error('MOC_gr_ind exceeds the available CM4 latitude grid.');
end
lat_gr(16);
lat_gr(155);
[RHO_PSI_interp,LAT_PSI_interp ] = meshgrid(rho2_interp,lat_gr(MOC_gr_ind ));


% Define the sub-region of interest for the source grid.
ind_gn = [11:245]; %from 75S to 65N
if ind_gn(end) > numel(lat_psi)
    error('ind_gn exceeds the available ACCESS latitude grid.');
end
lat_psi(11);
lat_psi(245);


LAT_PSI_sub = LAT_PSI(ind_gn,:);
RHO_PSI_sub= RHO_PSI(ind_gn,:);
temp = Psi_ASMOC(ind_gn,:,1);
F = scatteredInterpolant(LAT_PSI_sub(:), RHO_PSI_sub(:), temp(:), 'nearest', 'none');
% Preallocate the output array.
numTimeSteps = size(Psi_ASMOC, 3);
Psi_ASMOC_interp = NaN(size(LAT_PSI_interp,1), size(LAT_PSI_interp,2), numTimeSteps);
% Loop over time, updating the interpolant's Values property.
for t = 1:numTimeSteps
    % Display the current iteration (optional)
    % disp(t);
    % Update the interpolant's data for the current time slice.
    temp = Psi_ASMOC(ind_gn,:,t);
    F.Values = temp(:);
    % Evaluate the interpolant at the fixed query points.
    Psi_ASMOC_interp (:,:,t) = F(LAT_PSI_interp, RHO_PSI_interp);
end




test1 = squeeze(mean(Psi_ASMOC(ind_gn,and(rho2_i>1035,rho2_i<1037.2),:),3))/1e6;
test2 = squeeze(mean(Psi_ASMOC_interp,3))/1e6;
% figure
% hold on
% plot(LAT_PSI(ind_gn,1),test1(:,end))
% plot(lat_gr(MOC_gr_ind),test2(:,end))

% figure
% pcolor(LAT_PSI_interp,-RHO_PSI_interp,squeeze(mean(Psi_ASMOC_interp,3))/1e6)
% xlim([-75 65])
% ylim([-1037.2 -1035])
% clim([-20 20])
% cmocean('red',100)


LAT_ASMOC_interp = LAT_PSI_interp;
RHO_ASMOC_interp = RHO_PSI_interp;




save(fullfile(outputDir, ['FullDepth_ASMOC_interp_gr_r' realization '.mat']), ...
    'Psi_ASMOC_interp','LAT_ASMOC_interp','RHO_ASMOC_interp', ...
    'time_Psi','-v7.3')

end
    
