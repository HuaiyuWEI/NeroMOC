clearvars
close all
clc

toolboxRoot = 'D:\OneDrive - University of California\MATLAB toolboxs';
dataRoot = 'E:\Data_CMIP6\ACCESS_historical';
outputDir = dataRoot;  % Change this if you want figures written elsewhere.

addRequiredPath(toolboxRoot, 'MATLAB toolbox root');
ensureExistingFolder(dataRoot, 'ACCESS historical data directory');
ensureExistingFolder(outputDir, 'Figure output directory');

fontsize = 18;
LW = 2;


%% Plot OBP
obpData = loadRequiredMat(fullfile(dataRoot, 'Mascon_V5_OBP_r1.mat'), 'OBP mascon file');
obpField = obpData.Basin_id .* ...
    (obpData.OBP_2004_2009_mean + squeeze(mean(obpData.Input_vars_mascon(:, 1:24), 2)));
obpGrid = masconValuesToGrid(obpData.mascon_ID, obpData.mascon_ID_uniq, obpField);

obpColormap = cbrewer('seq', 'Blues', 20);
obpFigure = plotMasconSchematic( ...
    obpData.lon_mascon, obpData.lat_mascon, obpGrid, [0 6e7], 0:3e7:6e7, ...
    obpColormap, 'Ocean bottom pressure (Pa)', [0.37 0.24 0.01229 0.2], fontsize, LW);
exportgraphics(obpFigure, fullfile(outputDir, 'Schem_obp.png'), 'Resolution', 200);


%% Plot SSH
sshData = loadRequiredMat(fullfile(dataRoot, 'Mascon_V5_SSH_r1.mat'), 'SSH mascon file');
sshField = sshData.Basin_id .* ...
    (sshData.SSH_2004_2009_mean + squeeze(mean(sshData.Input_vars_mascon(:, 1:24), 2)));
sshField = sshField - mean(sshField, 'omitnan');
sshGrid = masconValuesToGrid(sshData.mascon_ID, sshData.mascon_ID_uniq, sshField);

sshColormap = cmocean('bal', 41);
sshFigure = plotMasconSchematic( ...
    sshData.lon_mascon, sshData.lat_mascon, sshGrid, [-2 2], -2:1:2, ...
    sshColormap, 'Sea surface height (m)', [0.385 0.25 0.01229 0.2], fontsize, LW);
exportgraphics(sshFigure, fullfile(outputDir, 'Schem_ssh.png'), 'Resolution', 200);


%% Plot zonal wind speed
uasData = loadRequiredMat(fullfile(dataRoot, 'Mascon_V5_UAS_r1.mat'), 'UAS mascon file');
uasField = uasData.Basin_id .* squeeze(mean(uasData.Input_vars_mascon(:, 1:24), 2));
uasGrid = masconValuesToGrid(uasData.mascon_ID, uasData.mascon_ID_uniq, uasField);

uasColormap = cmocean('cur', 41);
uasFigure = plotMasconSchematic( ...
    uasData.lon_mascon, uasData.lat_mascon, uasGrid, [-10 10], -10:5:10, ...
    uasColormap, 'Zonal wind speed (m/s)', [0.385 0.25 0.01229 0.2], fontsize, LW);
exportgraphics(uasFigure, fullfile(outputDir, 'Schem_uas.png'), 'Resolution', 200);


%% Schem_MOCs with nonlinear sigma2 axis
mocData = loadRequiredMat(fullfile(dataRoot,'ASMOC', 'MOC_r36_r40.mat'), ...
    'Full-depth ASMOC interpolation file');


fontname = 'Arial';
deg = char(176);
bg_gray = [0.90 0.90 0.90];
axLW = max(0.5, LW-0.5);
SmoothParam = 5;
SmoothMethod = 'moving';

% Nonlinear sigma2 axis settings
yBreak = 36.8;
stretchFactor = 4;
transformSigma2 = @(y) y + (y > yBreak) .* (y - yBreak) .* (stretchFactor - 1);

PsiMean = squeeze(mean(mocData.MOC_LPF_ALL, 1));
sigma2Levels = mocData.rho2_full - 1000;
[~, rhoIndexMax] = max(PsiMean, [], 2);
[~, rhoIndexMin] = min(PsiMean, [], 2);

rhoIndexMax = smooth(mocData.lat_psi(:), double(rhoIndexMax), SmoothParam, SmoothMethod);
rhoIndexMin = smooth(mocData.lat_psi(:), double(rhoIndexMin), SmoothParam, SmoothMethod);
rhoIndexMax = clipRhoIndices(round(rhoIndexMax), numel(sigma2Levels));
rhoIndexMin = clipRhoIndices(round(rhoIndexMin), numel(sigma2Levels));

rhoLineMax = sigma2Levels(rhoIndexMax);
rhoLineMin = sigma2Levels(rhoIndexMin);
rhoLineMax(mocData.lat_psi < -54.5) = nan;
rhoLineMin(mocData.lat_psi < -73.5) = nan;
[sigma2Real, LAT_plot] = meshgrid(sigma2Levels, mocData.lat_psi);

sigma2Plot = transformSigma2(sigma2Real);

framepos = [50 50 1300 380];
axpos = zeros(2, 4);
axpos(1, :) = [0.08 0.15 0.22 0.72];
axpos(2, :) = [0.315 0.15 0.57 0.72];
cbpos = [0.895 0.15 0.015 0.72];

mocFigure = figure('Position', framepos, 'Color', 'w', 'Renderer', 'painters');
cmp = flipud(cbrewer('div', 'RdBu', 51));
climPlot = [-25 25];
ylimPlot = [35.05 transformSigma2(37.18)];
yTicksReal = [35.2 36.0 36.8 36.9 37.0 37.1 37.2];
yTickLabels = arrayfun(@(v) sprintf('%.1f', v), yTicksReal, 'UniformOutput', false);

leftAxis = axes('Position', axpos(1, :));
hold(leftAxis, 'on');
plotMocPanel(leftAxis, LAT_plot, sigma2Plot, PsiMean, cmp, climPlot, ...
    [-74 -34], ylimPlot, transformSigma2(yTicksReal), yTickLabels, ...
    -75:5:-35, getSouthernLatitudeLabels(-75:5:-35, deg), fontsize, fontname, axLW, bg_gray);
leftAxis.TickLength = [0.008667*2 0];
plot(leftAxis, mocData.lat_psi, transformSigma2(rhoLineMax), '--k', 'LineWidth', axLW + 0.5);
plot(leftAxis, mocData.lat_psi, transformSigma2(rhoLineMin), '--k', 'LineWidth', axLW + 0.5);
ylabel(leftAxis, 'Density \sigma_2 (kg/m^3)', ...
    'FontSize', fontsize , 'Interpreter', 'tex', 'FontName', fontname);
text(leftAxis, 0.02, 0.98, 'SMOC', 'Units', 'normalized', ...
    'FontName', fontname, 'FontSize', fontsize, 'FontWeight', 'bold', ...
    'Interpreter', 'none', 'VerticalAlignment', 'top', 'Color', [0.1 0.1 0.1]);

rightAxis = axes('Position', axpos(2, :));
hold(rightAxis, 'on');
plotMocPanel(rightAxis, LAT_plot, sigma2Plot, PsiMean, cmp, climPlot, ...
    [-34 64.5], ylimPlot, transformSigma2(yTicksReal), repmat({''}, size(yTickLabels)), ...
    -35:5:75, getAtlanticLatitudeLabels(-35:5:75, deg), fontsize, fontname, axLW, bg_gray);
rightAxis.TickLength = [0.004*2 0];
plot(rightAxis, mocData.lat_psi, transformSigma2(rhoLineMax), '--k', 'LineWidth', axLW + 0.5);
text(rightAxis, 0.02, 0.98, 'AMOC', 'Units', 'normalized', ...
    'FontName', fontname, 'FontSize', fontsize, 'FontWeight', 'bold', ...
    'Interpreter', 'none', 'VerticalAlignment', 'top', 'Color', [0.1 0.1 0.1]);

cb = colorbar(leftAxis, 'Position', cbpos, 'LineWidth', axLW, 'FontSize', fontsize - 1);
cb.TickDirection = 'out';
cb.Box = 'off';
cb.FontName = fontname;
title(cb, 'Sv', 'FontSize', fontsize - 1, 'FontName', fontname, 'Interpreter', 'tex');

exportgraphics(mocFigure, fullfile(outputDir, 'Schem_MOCs_V2A.png'), 'Resolution', 300);


function figureHandle = plotMasconSchematic(lonMascon, latMascon, fieldGrid, climRange, ...
    colorbarTicks, colormapData, titleText, colorbarPosition, fontsize, lineWidth)
figureHandle = figure('Position', [50 0 800 500], 'Color', 'white');

m_proj('Azimuthal Equal-area', 'lat', -40, 'long', -25, 'radius', 110, 'rot', 20);
m_pcolor([lonMascon(:, 1) - 0.5 lonMascon], [latMascon(:, 1) latMascon], ...
    [fieldGrid(:, 1) fieldGrid]);
shading flat
m_coast('patch', [.7 .7 .7], 'edgecolor', 'none');
m_grid('xticklabel', [], 'yticklabel', [], 'linestyle', '--', ...
    'ytick', -60:30:60, 'linewidth', lineWidth - 1);

set(gca, 'FontSize', fontsize, 'LineWidth', lineWidth - 1, 'Layer', 'top');
box on
clim(climRange);
colormap(colormapData);

colorbar('ticks', colorbarTicks, ...
    'FontSize', fontsize - 2, ...
    'LineWidth', lineWidth - 1, ...
    'location', 'west', ...
    'Position', colorbarPosition, ...
    'AxisLocation', 'out');
title(titleText, 'FontSize', fontsize + 10);
end


function plotMocPanel(ax, latGrid, sigma2Plot, psiMean, colormapData, climPlot, xLimits, ...
    yLimits, yTickValues, yTickLabels, xTickValues, xTickLabels, fontsize, fontname, axLW, bgGray)
fieldPlot = pcolor(ax, latGrid, sigma2Plot, psiMean);
set(fieldPlot, 'EdgeColor', 'none');
shading(ax, 'interp');

colormap(ax, colormapData);
clim(ax, climPlot);
set(ax, 'YDir', 'reverse');

set(ax, ...
    'FontName', fontname, ...
    'FontSize', fontsize - 1, ...
    'LineWidth', axLW, ...
    'TickDir', 'out', ...
    'Box', 'off', ...
    'Layer', 'top', ...
    'XMinorTick', 'on', ...
    'YMinorTick', 'on', ...
    'XTickLabelRotation', 0, ...
    'TickLabelInterpreter', 'none');

ax.Color = bgGray;
axis(ax, [xLimits yLimits]);
xticks(ax, xTickValues);
xticklabels(ax, xTickLabels);
yticks(ax, yTickValues);
yticklabels(ax, yTickLabels);
end


function gridField = masconValuesToGrid(masconIdGrid, uniqueMasconIds, values)
gridField = nan(size(masconIdGrid));
for idx = 1:numel(values)
    gridField(masconIdGrid == uniqueMasconIds(idx)) = values(idx);
end
end


function labels = getSouthernLatitudeLabels(ticks, deg)
labels = arrayfun(@(~) '', ticks, 'UniformOutput', false);
labels(ticks == -70) = {sprintf('70%cS', deg)};
labels(ticks == -60) = {sprintf('60%cS', deg)};
labels(ticks == -50) = {sprintf('50%cS', deg)};
labels(ticks == -40) = {sprintf('40%cS', deg)};
end


function labels = getAtlanticLatitudeLabels(ticks, deg)
labels = repmat({''}, size(ticks));
for idx = 1:numel(ticks)
    value = ticks(idx);
    if value == 0
        labels{idx} = sprintf('0%c', deg);
    elseif mod(abs(value), 10) == 0
        if value < 0
            labels{idx} = sprintf('%d%cS', abs(value), deg);
        else
            labels{idx} = sprintf('%d%cN', value, deg);
        end
    end
end
end


function dataStruct = loadRequiredMat(filePath, label)
if ~isfile(filePath)
    error('%s not found:\n%s', label, filePath);
end
dataStruct = load(filePath);
end


function addRequiredPath(folderPath, label)
if ~isfolder(folderPath)
    error('%s not found:\n%s', label, folderPath);
end
addpath(genpath(folderPath));
end


function ensureExistingFolder(folderPath, label)
if ~isfolder(folderPath)
    error('%s not found:\n%s', label, folderPath);
end
end


function rhoIndex = clipRhoIndices(rhoIndex, nLevels)
rhoIndex = max(rhoIndex, 1);
rhoIndex = min(rhoIndex, nLevels);
end
