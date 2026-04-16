% Huaiyu Wei
% This script computes the 2004–2009 mean of model output.
% This is because the GRACE ocean bottom pressure data are anomalies
% relative to the Jan 2004–Dec 2009 baseline mean.
% Therefore, we compute the same baseline mean from the model output (OBP and SSH).


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


%% load the Mascon grid
grdfile = fullfile(projectRoot, 'BasinMask', 'Mascon_AtlSO.mat');
if ~isfile(grdfile)
    error('Mascon grid file not found: %s', grdfile);
end
load(grdfile)


%% User settings
ensembleStart = 1;
ensembleEnd = 35;
DataPath_ESM1d5 = 'E:\Data_CMIP6\ACCESS_historical\';
outputFile = sprintf('Mascon_TimeMean_2004_2009_r%d-r%d.mat', ensembleStart, ensembleEnd);

if ensembleStart > ensembleEnd
    error('ensembleStart must be less than or equal to ensembleEnd.')
end
if ~isfolder(DataPath_ESM1d5)
    error('Input data folder not found: %s', DataPath_ESM1d5);
end

%% load input variables from ACCESS-ESM1.5


InputNames = {'pbo','zos'};  % pbo: ocean bottom pressure % zos: sea surface height
NInputs = length(InputNames);
cd(DataPath_ESM1d5)

[filelist, r_numbers] = findAccessRealizationFiles( ...
    DataPath_ESM1d5, 'pbo_Omon_ACCESS-ESM1-5*.nc', 'ACCESS-ESM1-5 pbo');

numRealizations = ensembleEnd - ensembleStart + 1;
inputVarsByRealization = cell(numRealizations, 1);
for r_ind = ensembleStart:ensembleEnd

    [~, file_base, ~] = getAccessRealizationFile(filelist, r_numbers, r_ind);
    fprintf('Processing realization file: %s\n', file_base);

    Input_vars_temp = [];
    %%% Load input variables and apply missing value mask
    for var_ind = 1:NInputs
        InputName = InputNames{var_ind};
        file_name = fullfile(DataPath_ESM1d5, [InputName, file_base(4:end)]);
        Var_time_temp = ncread(file_name,'time');
        time_datetime = datetime(1850, 1, 1) + days(Var_time_temp);
        % Extract year and month
        time_year = year(time_datetime);
        time_month = month(time_datetime);
        % Find indices within Jan 2004 to Dec 2009
        ind_time = find( time_year >= 2004 & time_year <= 2009 );
        t_start = ind_time(1);
        t_end   = ind_time(end);
        t_count = t_end - t_start + 1;
        if(t_count~=72)
            error('check avg index')
        end
        info = ncinfo(file_name, InputName);
        var_size = info.Size;
        if isempty(Input_vars_temp)
            Input_vars_temp = nan(var_size(1), var_size(2), t_count, NInputs);
        end
        start = [1, 1, t_start];           % start at first lon, first lat, desired time index
        count = [var_size(1), var_size(2), t_count];  % full lon, full lat, subset of time
        % Read only the required subset
        Input_vars_temp(:,:,1:t_count,var_ind) = ncread(file_name, InputName, start, count);

    end
    inputVarsByRealization{r_ind - ensembleStart + 1} = Input_vars_temp;
end

Input_vars = cat(3, inputVarsByRealization{:});




%%% compute time-avg
Input_vars_avg = squeeze(mean(Input_vars,3));


% pcolor(squeeze(Input_vars(:,:,1))');shading flat
%%% put Atlantic in the center 
Input_vars_avg = [Input_vars_avg(101:360,:,:); Input_vars_avg(1:100,:,:)];
% figure
% pcolor(squeeze(Input_vars_avg(:,:,1))');shading flat


%%% Load coordinates and time (only once)
sample_file = fullfile(DataPath_ESM1d5, ['pbo', file_base(4:end)]);
Lat_ESM = ncread(sample_file, 'latitude');
Lon_ESM = ncread(sample_file, 'longitude');
Lon_ESM = [Lon_ESM(101:360,:); Lon_ESM(1:100,:)];
Lat_ESM = [Lat_ESM(101:360,:); Lat_ESM(1:100,:)];
Lon_ESM(Lon_ESM>180) = Lon_ESM(Lon_ESM>180)-360;

% figure
% pcolor(Lon_ESM,Lat_ESM,squeeze(Input_vars_avg(:,:,2))); shading flat


% Average data from ACCESS's grid onto the mascons
Input_vars_mascon = averageFieldToMascons(Input_vars_avg, Lon_ESM, Lat_ESM, ...
    lon_mascon_bound1, lon_mascon_bound2, lat_mascon_bound1, ...
    lat_mascon_bound2, flag_across_180);

figure
scatter(lon_mascon_center, lat_mascon_center,15,Input_vars_mascon(:,1),'filled')
title('Ocean bottom pressure Jan 2004–Dec 2009 mean')
shading flat
colorbar

figure
scatter(lon_mascon_center, lat_mascon_center,15,Input_vars_mascon(:,2),'filled')
title('Sea surface height Jan 2004–Dec 2009 mean')
shading flat
colorbar

%%% Save result

save(outputFile, ...
    'lon_mascon','lat_mascon', 'Input_vars_mascon','Basin_id', ...
    'lon_mascon_center','lat_mascon_center','mascon_ID','mascon_ID_uniq', ...
    'lon_mascon_bound1','lon_mascon_bound2','lat_mascon_bound1','lat_mascon_bound2','ind_time','-v7.3');





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

