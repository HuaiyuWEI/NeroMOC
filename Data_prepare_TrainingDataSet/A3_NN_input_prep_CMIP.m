% Huaiyu Wei
% Interpolate ACCESS-ESM1-5 model output from the native grid to the JPL
% GRACE mascon grid. Variables include ocean bottom pressure, sea surface
% height, and zonal near-surface wind speed.

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
grdfile = fullfile(projectRoot, 'BasinMask', 'Mascon_AtlSO.mat');
if ~isfile(grdfile)
    error('Mascon grid file not found: %s', grdfile);
end
load(grdfile)

%% User settings
ensembleStart = 1;
ensembleEnd = 35;
baselineMeanFile = 'E:\Data_CMIP6\ACCESS_historical\Mascon_TimeMean_2004_2009_r1-r35.mat';
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_historical\';
DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP585\';
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP245\';
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP370\';
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP126\';
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP126\2100-2300\';
% DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_SSP585\2100-2300\';

if ensembleStart > ensembleEnd
    error('ensembleStart must be less than or equal to ensembleEnd.')
end
if ~isfile(baselineMeanFile)
    error('Baseline mean file not found: %s', baselineMeanFile);
end
if ~isfolder(DataPath_ESM1d5)
    error('Input data folder not found: %s', DataPath_ESM1d5);
end

%% load reference OBP and SSH

temp = load(baselineMeanFile,'Input_vars_mascon');
OBP_2004_2009_mean = temp.Input_vars_mascon(:,1);
SSH_2004_2009_mean = temp.Input_vars_mascon(:,2);

%%  load input variables from ACCESS-ESM1.5
%%% ocean bottom pressure and sea surface height are outputed in the same
%%% grid in ACCESS-ESM1.5, so we can deal with them in one loop.

InputNames = {'pbo','zos'};
NInputs = length(InputNames);
cd(DataPath_ESM1d5)

[filelist, r_numbers] = findAccessRealizationFiles( ...
    DataPath_ESM1d5, 'pbo_Omon_ACCESS-ESM1-5*.nc', 'ACCESS-ESM1-5 pbo');



sample_file = fullfile(DataPath_ESM1d5,  filelist{1});
ncdisp(sample_file)
Lat_ESM = ncread(sample_file, 'latitude');
Lon_ESM = ncread(sample_file, 'longitude');
Lon_ESM = [Lon_ESM(101:360,:); Lon_ESM(1:100,:)];
Lat_ESM = [Lat_ESM(101:360,:); Lat_ESM(1:100,:)];
Lon_ESM(Lon_ESM>180) = Lon_ESM(Lon_ESM>180)-360;
[Nlon,Nlat] = size(Lon_ESM);

Nsamps = size(ncread(sample_file, 'time')); Nsamps= Nsamps(1);



for r_ind = ensembleStart:ensembleEnd


    [fileIndices, file_base, realization] = getAccessRealizationFile( ...
        filelist, r_numbers, r_ind, true);

    Input_vars = [];
    Input_time  = [];

    for i = fileIndices
        fprintf('Processing realization file: %s\n', filelist{i});

        Input_vars_temp = nan(Nlon,Nlat,Nsamps,NInputs);
        %%% Load input variables and apply missing value mask
        for var_ind = 1:NInputs
            InputName = InputNames{var_ind};
            file_name = fullfile(DataPath_ESM1d5, [InputName, filelist{i}(4:end)]);
            Input_vars_temp(:,:,:,var_ind) = ncread(file_name,InputName);
            Var_time_temp = ncread(file_name,'time');
        end

        Input_vars = cat(3,Input_vars,Input_vars_temp);
        Input_time = cat(1,Input_time,Var_time_temp);
    end

    if(any(diff(Input_time)<0))
        error('Check time')
    end
    Nsamps_ESM1d5 = length(Input_time);
    % Input_vars [lon x lat x time x var]
    % pcolor(squeeze(Input_vars(:,:,1))');shading flat
    %%% put Atlantic in the center
    Input_vars = [Input_vars(101:360,:,:,:); Input_vars(1:100,:,:,:)];
    % pcolor(squeeze(Input_vars(:,:,1))');shading flat


    % Average OBP from ACCESS's grid onto the mascons
    for ii = 1:NInputs
        Input_vars_mascon = averageFieldToMascons(Input_vars(:,:,:,ii), Lon_ESM, Lat_ESM, ...
            lon_mascon_bound1, lon_mascon_bound2, lat_mascon_bound1, ...
            lat_mascon_bound2, flag_across_180);


        %%% Save result
        if(ii == 1)
            Input_vars_mascon = Input_vars_mascon - OBP_2004_2009_mean;
            save(['Mascon_V5_OBP_r' realization '.mat'], ...
                'lon_mascon','lat_mascon', 'Input_vars_mascon','Basin_id','OBP_2004_2009_mean', ...
                'lon_mascon_center','lat_mascon_center','mascon_ID','mascon_ID_uniq', 'flag_across_180', ...
                'lon_mascon_bound1','lon_mascon_bound2','lat_mascon_bound1','lat_mascon_bound2','Input_time','-v7.3');
        elseif(ii == 2)
            Input_vars_mascon = Input_vars_mascon - SSH_2004_2009_mean;
            save(['Mascon_V5_SSH_r' realization '.mat'], ...
                'lon_mascon','lat_mascon', 'Input_vars_mascon','Basin_id', 'flag_across_180', ...
                'lon_mascon_center','lat_mascon_center','mascon_ID','mascon_ID_uniq','SSH_2004_2009_mean', ...
                'lon_mascon_bound1','lon_mascon_bound2','lat_mascon_bound1','lat_mascon_bound2','Input_time','-v7.3');
        else
            error('Saving data failed.')
        end
    end
    %%% Clean up for next loop
    clear Input_vars Input_vars_mascon

end



%% wind speed

InputNames = {'uas'};
NInputs = length(InputNames);
cd(DataPath_ESM1d5)

[filelist, r_numbers] = findAccessRealizationFiles( ...
    DataPath_ESM1d5, 'uas_Amon_ACCESS-ESM1-5*.nc', 'ACCESS-ESM1-5 uas');


sample_file = fullfile(DataPath_ESM1d5, filelist{1});
Lat_atm = ncread(sample_file, 'lat');
Lon_atm = ncread(sample_file, 'lon');
[Lat_atm,Lon_atm] = meshgrid(Lat_atm,Lon_atm);
Lon_atm = [Lon_atm(97:end,:); Lon_atm(1:96,:)];
Lat_atm = [Lat_atm(97:end,:); Lat_atm(1:96,:)];
Lon_atm(Lon_atm>180) = Lon_atm(Lon_atm>180)-360;
Nsamps = size(ncread(sample_file, 'time')); Nsamps= Nsamps(1);

[Nlon_atm,Nlat_atm] = size(Lon_atm);


for r_ind = ensembleStart:ensembleEnd


    [fileIndices, file_base, realization] = getAccessRealizationFile( ...
        filelist, r_numbers, r_ind, true);

    Input_vars = [];
    Input_time  = [];

    for i = fileIndices
        fprintf('Processing realization file: %s\n', filelist{i});

        Input_vars_temp = nan(Nlon_atm,Nlat_atm,Nsamps,NInputs);
        %%% Load input variables and apply missing value mask
        for var_ind = 1:NInputs
            InputName = InputNames{var_ind};
            file_name = fullfile(DataPath_ESM1d5, filelist{i});
            Input_vars_temp(:,:,:,var_ind) = ncread(file_name,InputName);
            Var_time_temp = ncread(file_name,'time');
        end

        Input_vars = cat(3,Input_vars,Input_vars_temp);
        Input_time = cat(1,Input_time,Var_time_temp);
    end

    if(any(diff(Input_time)<0))
        error('Check time')
    end
    Nsamps_ESM1d5 = length(Input_time);
    % Input_vars [lon x lat x time x var]

  

    % pcolor(squeeze(Input_vars(:,:,1))');shading flat
    %%% put Atlantic in the center
    Input_vars = [Input_vars(97:end,:,:,:); Input_vars(1:96,:,:,:)];
    % pcolor(squeeze(Input_vars(:,:,1))');shading flat

    % figure
    % pcolor(Lon_atm,Lat_atm,squeeze(Input_vars(:,:,88,1))); shading flat


    % Average tauu from ACCESS's grid onto the mascons
    Input_vars_mascon = averageFieldToMascons(Input_vars, Lon_atm, Lat_atm, ...
        lon_mascon_bound1, lon_mascon_bound2, lat_mascon_bound1, ...
        lat_mascon_bound2, flag_across_180);

    % figure
    % scatter(lon_mascon_center, lat_mascon_center,15,Input_vars_mascon(:,88,1),'filled')
    % shading flat
    % colorbar


    %%% Save result
    if(strcmp(InputName,'uas'))
        save(['Mascon_V5_UAS_r' realization '.mat'], ...
            'lon_mascon','lat_mascon', 'Input_vars_mascon','Basin_id', 'flag_across_180',...
            'lon_mascon_center','lat_mascon_center','mascon_ID','mascon_ID_uniq', ...
            'lon_mascon_bound1','lon_mascon_bound2','lat_mascon_bound1','lat_mascon_bound2','Input_time','-v7.3');
    else
        error('Saving data failed.')
    end

    %%% Clean up for next loop
    clear Input_vars Input_vars_mascon

end






%% Local helper functions
function [filelist, realizationNumbers] = findAccessRealizationFiles(dataPath, filePattern, fileLabel)
% Find ACCESS-ESM1-5 files and parse their realization numbers.

files = dir(fullfile(dataPath, filePattern));
filelist = {files.name};
disp(filelist')

if isempty(filelist)
    error('No %s files were found in %s', fileLabel, dataPath);
end

realizationNumbers = zeros(size(filelist));
for i = 1:numel(filelist)
    tokens = regexp(filelist{i}, '_r(\d+)i', 'tokens');
    if ~isempty(tokens)
        realizationNumbers(i) = str2double(tokens{1}{1});
    end
end
end

function [matchIndices, fileBase, realization] = getAccessRealizationFile( ...
    filelist, realizationNumbers, realizationIndex, allowMultiple)
% Return matching file indices and the realization tag for one ensemble member.

if nargin < 4
    allowMultiple = false;
end

matches = find(realizationNumbers == realizationIndex);
disp(['Number of files in realization r' num2str(realizationIndex) ': ' num2str(numel(matches))])

if isempty(matches)
    error('Missing file for realization r%d.', realizationIndex);
end
if ~allowMultiple && numel(matches) > 1
    error('Multiple files found for realization r%d. Please check the input directory.', realizationIndex);
end

matchIndices = matches;
fileBase = filelist{matches(1)};

tokens = regexp(fileBase, 'r(\d+)i1p1f1', 'tokens');
realization = '000';
if ~isempty(tokens)
    realization = tokens{1}{1};
end
end
