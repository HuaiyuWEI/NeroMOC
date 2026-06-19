%%% Define MOC strength as the MOC variability at density level where
%%% timemean MOC is maximized

clearvars
close all
clc

toolboxRoot = 'D:\OneDrive - University of California\MATLAB toolboxs';
legacyCodeRoot = 'D:\OneDrive - University of California\MATLAB Codes\CMIP6';
scriptDir = fileparts(mfilename('fullpath'));
polishedScriptsRoot = fileparts(scriptDir);
projectRoot = fileparts(polishedScriptsRoot);
if isfolder(legacyCodeRoot)
    codeRoot = legacyCodeRoot;
else
    codeRoot = projectRoot;
end
basedir = 'E:\Analysis2026\ACCESS_hist+SSP585\ASMOC\results_LPF2Year';

addRequiredPath(toolboxRoot, 'MATLAB toolbox root');
addRequiredPath(codeRoot, 'CMIP6 code root');

SmoothParam = 5;
SmoothMethod = 'moving';

%%  load baseline model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cmipName = 'SSP245';
covariate_name = 'obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5';
NNname = 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
referenceNNnameForStrength = 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_dir_baseline = fullfile(basedir, NNname, covariate_name);
reference_model_dir = fullfile(basedir, referenceNNnameForStrength, covariate_name);
monteCarloDir = fullfile(model_dir_baseline, 'MonteCarlo');

ensureExistingFolder(model_dir_baseline, 'Baseline model directory');
ensureExistingFolder(reference_model_dir, 'Reference model directory for MOC strength');
ensureExistingFolder(monteCarloDir, 'Monte Carlo results directory');


%%% use MonteCarlo Results to compute correlation
load(resolveExistingMatFile(fullfile(monteCarloDir, ...
    ['R2_' cmipName '_PNoise40_SSHNoise1_MC500']), 'Monte Carlo R2 results'));
[Nsamps,Nrho,Nlat]= size(pred_mean_yz);
truth = reshape(Psi,[length(Psi),140,18]); 
pred = permute(pred_mean_yz, [1,3,2]);
rmse = squeeze(sqrt(mean((truth - pred).^2)))';
save(fullfile(monteCarloDir, ['rmse_' cmipName '_PNoise40_SSHNoise1_MC500.mat']), "rmse")
truth_std = squeeze(std(truth,1))';

uncertainty = sqrt(permute(pred_variance_yz, [1,3,2]));
r2_baseline = nan(Nrho,Nlat);
for j = 1:Nlat
    for k = 1:Nrho
        r2_baseline(k,j) = (corr(truth(:,j,k),pred(:,j,k))).^2;
    end
end



%%%%%%%%% r2 for detrended reconstruction and diagnostic
N_realization = 5;
if mod(Nsamps, N_realization) ~= 0
    error('Expected Nsamps (%d) to be divisible by N_realization (%d).', Nsamps, N_realization);
end
Nsamps_r = Nsamps/N_realization;

r2_detrend = nan(Nrho,Nlat);
truth_detrend = nan*truth;
pred_detrend = nan*truth;
for j = 1:Nlat
    for k = 1:Nrho
        for i = 1:N_realization
            ind_start = (i-1)*Nsamps_r+1;
            ind_end =i*Nsamps_r;
            truth_detrend(ind_start:ind_end,j,k) = detrend(truth(ind_start:ind_end,j,k));
            pred_detrend(ind_start:ind_end,j,k) = detrend(pred(ind_start:ind_end,j,k));
        end
        r2_detrend(k,j) = (corr(truth_detrend(:,j,k),pred_detrend(:,j,k))).^2; 
    end
end

rho2 = rho2-1000;


%%% Time mean value of diagnosed MOC; and define MOC strength
referencePrediction = load(resolveExistingMatFile(fullfile(reference_model_dir, ...
    ['Pred_' cmipName]), 'Reference prediction for MOC strength'));
y = referencePrediction.y;
Nsample = length(y);
MOC= reshape(y,[Nsample,Nlat,Nrho]);
MOC_mean = squeeze(mean(MOC))';
MOC_std = squeeze(std(MOC))';
% mid-depth cell
[~,rho_ind_mean] = max(MOC_mean,[],1);
rho_ind_mean = smooth(lat_psi,rho_ind_mean,SmoothParam,SmoothMethod);
rho_ind_mean = round(rho_ind_mean);
rho_SSP = rho2(rho_ind_mean);
rho_SSP(lat_psi < -55.5) = nan;

% abyssal cell
[~,rho_ind_abys_mean] = min(MOC_mean,[],1);
rho_ind_abys_mean = smooth(lat_psi,rho_ind_abys_mean,SmoothParam,SmoothMethod);
rho_ind_abys_mean = round(rho_ind_abys_mean);
rho_SSP_abys = rho2(rho_ind_abys_mean);

%%% MOC strength
truth_strength_mid = zeros(Nsample,Nlat);
truth_strength_abys = zeros(Nsample,Nlat);
pred_strength_mid = zeros(Nsample,Nlat);
pred_strength_abys = zeros(Nsample,Nlat);
for jj = 1:Nlat
truth_strength_mid(:,jj) = truth(:,jj,rho_ind_mean(jj));
truth_strength_abys (:,jj) = truth(:,jj,rho_ind_abys_mean(jj));
pred_strength_mid(:,jj) = pred(:,jj,rho_ind_mean(jj));
pred_strength_abys (:,jj) = pred(:,jj,rho_ind_abys_mean(jj));
end
pred_strength_mid_ind = rho_ind_mean;
pred_strength_abys_ind = rho_ind_abys_mean;



% Find the latitude where MOC strength is maximized
[~, temp] = max(MOC_mean,[],'all');
[~, temp2] = ind2sub(size(MOC_mean), temp);
Lax_maxSOMOC_mid = lat_psi(temp2);

[~, temp] = min(MOC_mean,[],'all');
[~, temp2] = ind2sub(size(MOC_mean), temp);
Lax_maxSOMOC_abys = lat_psi(temp2);

AMOC_mean=MOC_mean;
AMOC_mean(:,1:41)=nan;
[~, temp] = max(AMOC_mean,[],'all');
[~, temp2] = ind2sub(size(AMOC_mean), temp);
Lax_maxAMOC = lat_psi(temp2);





%% plot the reconstruction skill of the baseline NN 

% ---- layout ----
framepos = [100 50 1300 600];
axpos = zeros(5,4);
axpos(1,:) = [0.08 0.2 0.25 0.6];
axpos(2,:) = [0.34 0.2 0.6  0.6];
cbpos      = [0.95 0.2 0.01 0.6];

% ---- style ----
fontsize = 17;
LW = 1.5;
fontname = 'Arial';
deg = char(176);

bg_gray = [0.90 0.90 0.90];     % NaN/background
dotSize = 8;

% ---- plot field ----
r2_plot = r2_baseline;
r2_plot(MOC_std < 0.5) = nan;

% ---- masks (compute once) ----
MaskNeg = (MOC_mean < 0) & (MOC_std > 0.5);   % [lat x rho] or [rho x lat] depending on your arrays
[JJneg, IIneg] = find(MaskNeg');              % JJneg->lat index, IIneg->rho index (match your earlier convention)

stepDots = 1;                                  % 1=all, 2=sparser
JJneg = JJneg(1:stepDots:end);
IIneg = IIneg(1:stepDots:end);

% ---- limits ----
yTop = (rho2(end)+rho2(end-1))/2;

% ---- colormap (0..1) ----
clim_plot = [0 1];
cmp = pmkmp(40,'Swtth');        % keep your preferred cmap
% alternative: cmp = turbo(256);

% ---- figure ----
figure('Position',framepos,'Color','w','Renderer','painters');

%%% ===================== LEFT: SOMOC =====================
ax1 = axes('Position',axpos(1,:)); hold(ax1,'on');

h = imagesc(ax1, lat_psi, rho2, r2_plot);
set(h,'AlphaData',~isnan(r2_plot));           % show only valid pixels
set(ax1,'YDir','reverse');
colormap(ax1, cmp);
clim(ax1, clim_plot);

axis(ax1, [-74 -34 35 yTop]);

% ticks/labels
xt = -75:5:-35;
xticks(ax1, xt);
lbl = arrayfun(@(x) '', xt, 'UniformOutput', false);
lbl(xt==-70) = {sprintf('70%cS',deg)};
lbl(xt==-60) = {sprintf('60%cS',deg)};
lbl(xt==-50) = {sprintf('50%cS',deg)};
lbl(xt==-40) = {sprintf('40%cS',deg)};
xticklabels(ax1, lbl);

yt = 35:0.5:37.5;
yticks(ax1, yt);
ylabel(ax1, 'Density \sigma_2 (kg/m^3)', ...
    'FontSize',fontsize,'Interpreter','tex','FontName',fontname);

% axis styling
set(ax1,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
    'TickDir','out','Box','off','Layer','top', ...
    'TickLength',[0.008667 0], ...
    'XTickLabelRotation',0, ...
    'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
ax1.Color = bg_gray;

% negative MOC dots
scatter(ax1, lat_psi(JJneg), rho2(IIneg), dotSize, 'w', 'filled', ...
    'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

% rho lines (most-frequent density of max)
plot(ax1, lat_psi, rho_SSP,      '--k', 'LineWidth', LW+0.5);
plot(ax1, lat_psi, rho_SSP_abys, '--k', 'LineWidth', LW+0.5);

% titles
% title(ax1, 'SMOC', 'FontSize',fontsize,'Interpreter','tex','FontName',fontname);

% colorbar (single, shared) — attach to left like your example
cb = colorbar(ax1,'Position',cbpos,'LineWidth',LW,'FontSize',fontsize);
cb.TickDirection = 'out';
cb.Box = 'off';
cb.FontName = fontname;
title(cb,'r^2','FontSize',fontsize,'FontName',fontname,'Interpreter','tex');

%%% ===================== RIGHT: AMOC =====================
ax2 = axes('Position',axpos(2,:)); hold(ax2,'on');

h = imagesc(ax2, lat_psi, rho2, r2_plot);
set(h,'AlphaData',~isnan(r2_plot));
set(ax2,'YDir','reverse');
colormap(ax2, cmp);
clim(ax2, clim_plot);

axis(ax2, [-34 64 35 yTop]);

% ticks/labels
xt = -35:5:75;
xticks(ax2, xt);

lbl = repmat({''}, size(xt));
for i = 1:numel(xt)
    v = xt(i);
    if v == 0
        lbl{i} = sprintf('0%c', deg);
    elseif mod(abs(v),10)==0
        if v < 0, lbl{i} = sprintf('%d%cS', abs(v), deg);
        else,     lbl{i} = sprintf('%d%cN', v, deg);
        end
    end
end
xticklabels(ax2, lbl);

yt = 35:0.5:37.5;
yticks(ax2, yt);
yticklabels(ax2, {''});

% axis styling
set(ax2,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
    'TickDir','out','Box','off','Layer','top', ...
    'TickLength',[0.004 0], ...
    'XTickLabelRotation',0, ...
    'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
ax2.Color = bg_gray;

% negative MOC dots
scatter(ax2, lat_psi(JJneg), rho2(IIneg), dotSize, 'w', 'filled', ...
    'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

% rho line
plot(ax2, lat_psi, rho_SSP, '--k', 'LineWidth', LW+0.5);

% title(ax2, 'AMOC', 'FontSize',fontsize,'Interpreter','tex','FontName',fontname);


text(ax1, 0.02, 0.98, 'SMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);

text(ax2, 0.02, 0.98, 'AMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);

%%% export
exportgraphics(gcf, fullfile(monteCarloDir, ['corr2_' cmipName '_v3.png']), 'Resolution', 300);
% exportgraphics(gcf, fullfile(monteCarloDir, ['corr2_' cmipName '.pdf']), 'ContentType','vector');




%% plot r2 of detrended prediction and reconstruction



% ---- layout ----
framepos = [100 50 1300 600];
axpos = zeros(5,4);
axpos(1,:) = [0.08 0.2 0.25 0.6];
axpos(2,:) = [0.34 0.2 0.6  0.6];
cbpos      = [0.95 0.2 0.01 0.6];

% ---- style ----
fontsize = 16;
LW = 1.5;
fontname = 'Arial';
deg = char(176);

bg_gray = [0.90 0.90 0.90];     % NaN/background
dotSize = 8;

% ---- plot field ----
r2_plot = r2_detrend;
r2_plot(MOC_std < 0.5) = nan;

% ---- masks (compute once) ----
MaskNeg = (MOC_mean < 0) & (MOC_std > 0.5);   % [lat x rho] or [rho x lat] depending on your arrays
[JJneg, IIneg] = find(MaskNeg');              % JJneg->lat index, IIneg->rho index (match your earlier convention)

stepDots = 1;                                  % 1=all, 2=sparser
JJneg = JJneg(1:stepDots:end);
IIneg = IIneg(1:stepDots:end);

% ---- limits ----
yTop = (rho2(end)+rho2(end-1))/2;

% ---- colormap (0..1) ----
clim_plot = [0 1];
cmp = pmkmp(40,'Swtth');        % keep your preferred cmap
% alternative: cmp = turbo(256);

% ---- figure ----
figure('Position',framepos,'Color','w','Renderer','painters');

%%% ===================== LEFT: SOMOC =====================
ax1 = axes('Position',axpos(1,:)); hold(ax1,'on');

h = imagesc(ax1, lat_psi, rho2, r2_plot);
set(h,'AlphaData',~isnan(r2_plot));           % show only valid pixels
set(ax1,'YDir','reverse');
colormap(ax1, cmp);
clim(ax1, clim_plot);

axis(ax1, [-74 -34 35 yTop]);

% ticks/labels
xt = -75:5:-35;
xticks(ax1, xt);
lbl = arrayfun(@(x) '', xt, 'UniformOutput', false);
lbl(xt==-70) = {sprintf('70%cS',deg)};
lbl(xt==-60) = {sprintf('60%cS',deg)};
lbl(xt==-50) = {sprintf('50%cS',deg)};
lbl(xt==-40) = {sprintf('40%cS',deg)};
xticklabels(ax1, lbl);

yt = 35:0.5:37.5;
yticks(ax1, yt);
ylabel(ax1, 'Density \sigma_2 (kg/m^3)', ...
    'FontSize',fontsize,'Interpreter','tex','FontName',fontname);

% axis styling
set(ax1,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
    'TickDir','out','Box','off','Layer','top', ...
    'TickLength',[0.008667 0], ...
    'XTickLabelRotation',0, ...
    'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
ax1.Color = bg_gray;

% negative MOC dots
scatter(ax1, lat_psi(JJneg), rho2(IIneg), dotSize, 'w', 'filled', ...
    'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

% rho lines (most-frequent density of max)
plot(ax1, lat_psi, rho_SSP,      '--k', 'LineWidth', LW+0.5);
plot(ax1, lat_psi, rho_SSP_abys, '--k', 'LineWidth', LW+0.5);

% titles
% title(ax1, 'SMOC', 'FontSize',fontsize,'Interpreter','tex','FontName',fontname);

% colorbar (single, shared) — attach to left like your example
cb = colorbar(ax1,'Position',cbpos,'LineWidth',LW,'FontSize',fontsize);
cb.TickDirection = 'out';
cb.Box = 'off';
cb.FontName = fontname;
title(cb,'r^2','FontSize',fontsize,'FontName',fontname,'Interpreter','tex');

%%% ===================== RIGHT: AMOC =====================
ax2 = axes('Position',axpos(2,:)); hold(ax2,'on');

h = imagesc(ax2, lat_psi, rho2, r2_plot);
set(h,'AlphaData',~isnan(r2_plot));
set(ax2,'YDir','reverse');
colormap(ax2, cmp);
clim(ax2, clim_plot);

axis(ax2, [-34 64 35 yTop]);

% ticks/labels
xt = -35:5:75;
xticks(ax2, xt);

lbl = repmat({''}, size(xt));
for i = 1:numel(xt)
    v = xt(i);
    if v == 0
        lbl{i} = sprintf('0%c', deg);
    elseif mod(abs(v),10)==0
        if v < 0, lbl{i} = sprintf('%d%cS', abs(v), deg);
        else,     lbl{i} = sprintf('%d%cN', v, deg);
        end
    end
end
xticklabels(ax2, lbl);

yt = 35:0.5:37.5;
yticks(ax2, yt);
yticklabels(ax2, {''});

% axis styling
set(ax2,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
    'TickDir','out','Box','off','Layer','top', ...
    'TickLength',[0.004 0], ...
    'XTickLabelRotation',0, ...
    'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
ax2.Color = bg_gray;

% negative MOC dots
scatter(ax2, lat_psi(JJneg), rho2(IIneg), dotSize, 'w', 'filled', ...
    'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

% rho line
plot(ax2, lat_psi, rho_SSP, '--k', 'LineWidth', LW+0.5);

% title(ax2, 'AMOC', 'FontSize',fontsize,'Interpreter','tex','FontName',fontname);


text(ax1, 0.02, 0.98, 'SMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);

text(ax2, 0.02, 0.98, 'AMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);

exportgraphics(gcf, fullfile(monteCarloDir, ['corr2_detrend_' cmipName '_v3.png']), 'resolution', 200);
%% plot the root mean squared error of the prediction
%%% and %% plot the std of the moc (truth)
framepos = [100 50 1300 600];
% figure('Position',framepos);


% ---- layout ----
axpos(1,:) = [0.08 0.55 0.15 0.32];
axpos(2,:) = [0.24 0.55 0.35 0.32];
axpos(3,:) = [0.08 0.10 0.15 0.32];
axpos(4,:) = [0.24 0.10 0.35 0.32];
cbpos      = [0.6 0.1 0.01 0.77];
% ---- style ----
fontsize = 12;
LW = 1.5;
fontname = 'Arial';
deg = char(176);

bg_gray = [0.90 0.90 0.90];     % NaN/background
dotSize = 8;

% ---- plot field ----
r2_plot = rmse;
% r2_plot(MOC_std < 0.5) = nan;

% ---- masks (compute once) ----
MaskNeg = (MOC_mean < 0);   % [lat x rho] or [rho x lat] depending on your arrays
[JJneg, IIneg] = find(MaskNeg');              % JJneg->lat index, IIneg->rho index (match your earlier convention)

stepDots = 1;                                  % 1=all, 2=sparser
JJneg = JJneg(1:stepDots:end);
IIneg = IIneg(1:stepDots:end);

% ---- limits ----
yTop = (rho2(end)+rho2(end-1))/2;

% ---- colormap (0..1) ----
clim_plot = [0 6];
cmp = cbrewer('div','Spectral',40);
cmp = flipud(cmp);   
% alternative: cmp = turbo(256);

% ---- figure ----
fh = figure('Position',framepos,'Color','w','Renderer','painters');

%%% ===================== LEFT: SOMOC =====================
ax1 = axes('Position',axpos(1,:)); hold(ax1,'on');

h = imagesc(ax1, lat_psi, rho2, r2_plot);
set(h,'AlphaData',~isnan(r2_plot));           % show only valid pixels
set(ax1,'YDir','reverse');
colormap(ax1, cmp);
clim(ax1, clim_plot);

axis(ax1, [-74 -34 35 yTop]);

% ticks/labels
xt = -75:5:-35;
xticks(ax1, xt);
lbl = arrayfun(@(x) '', xt, 'UniformOutput', false);
lbl(xt==-70) = {sprintf('70%cS',deg)};
lbl(xt==-60) = {sprintf('60%cS',deg)};
lbl(xt==-50) = {sprintf('50%cS',deg)};
lbl(xt==-40) = {sprintf('40%cS',deg)};
xticklabels(ax1, lbl);

yt = 35:0.5:37.5;
yticks(ax1, yt);
ylabel(ax1, 'Density \sigma_2 (kg/m^3)', ...
    'FontSize',fontsize,'Interpreter','tex','FontName',fontname);

% axis styling
set(ax1,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
    'TickDir','out','Box','off','Layer','top', ...
    'TickLength',[0.008667 0], ...
    'XTickLabelRotation',0, ...
    'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
ax1.Color = bg_gray;

% negative MOC dots
scatter(ax1, lat_psi(JJneg), rho2(IIneg), dotSize, 'w', 'filled', ...
    'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

% rho lines (most-frequent density of max)
plot(ax1, lat_psi, rho_SSP,      '--k', 'LineWidth', LW+0.5);
plot(ax1, lat_psi, rho_SSP_abys, '--k', 'LineWidth', LW+0.5);

% titles
% title(ax1, 'SMOC', 'FontSize',fontsize,'Interpreter','tex','FontName',fontname);

% colorbar (single, shared) — attach to left like your example
cb = colorbar(ax1,'Position',cbpos,'LineWidth',LW,'FontSize',fontsize);
cb.TickDirection = 'out';
cb.Box = 'off';
cb.FontName = fontname;
title(cb,'(Sv)','FontSize',fontsize,'FontName',fontname,'Interpreter','tex');
text(0.0, 1.1, '(A) DBNN reconstruction error', 'Units','normalized','FontSize',fontsize,'Interpreter','none')


%%% ===================== RIGHT: AMOC =====================
ax2 = axes('Position',axpos(2,:)); hold(ax2,'on');

h = imagesc(ax2, lat_psi, rho2, r2_plot);
set(h,'AlphaData',~isnan(r2_plot));
set(ax2,'YDir','reverse');
colormap(ax2, cmp);
clim(ax2, clim_plot);

axis(ax2, [-34 64 35 yTop]);

% ticks/labels
xt = -35:5:75;
xticks(ax2, xt);

lbl = repmat({''}, size(xt));
for i = 1:numel(xt)
    v = xt(i);
    if v == 0
        lbl{i} = sprintf('0%c', deg);
    elseif mod(abs(v),10)==0
        if v < 0, lbl{i} = sprintf('%d%cS', abs(v), deg);
        else,     lbl{i} = sprintf('%d%cN', v, deg);
        end
    end
end
xticklabels(ax2, lbl);

yt = 35:0.5:37.5;
yticks(ax2, yt);
yticklabels(ax2, {''});

% axis styling
set(ax2,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
    'TickDir','out','Box','off','Layer','top', ...
    'TickLength',[0.004 0], ...
    'XTickLabelRotation',0, ...
    'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
ax2.Color = bg_gray;

% negative MOC dots
scatter(ax2, lat_psi(JJneg), rho2(IIneg), dotSize, 'w', 'filled', ...
    'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

% rho line
plot(ax2, lat_psi, rho_SSP, '--k', 'LineWidth', LW+0.5);

% title(ax2, 'AMOC', 'FontSize',fontsize,'Interpreter','tex','FontName',fontname);


text(ax1, 0.7, 0.99, 'SMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);

text(ax2, 0.02, 0.99, 'AMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);






%%%%%%%%%%% STD
% ---- plot field ----
r2_plot = truth_std;
% r2_plot(MOC_std < 0.5) = nan;

% ---- masks (compute once) ----
MaskNeg = (MOC_mean < 0);   % [lat x rho] or [rho x lat] depending on your arrays
[JJneg, IIneg] = find(MaskNeg');              % JJneg->lat index, IIneg->rho index (match your earlier convention)

stepDots = 1;                                  % 1=all, 2=sparser
JJneg = JJneg(1:stepDots:end);
IIneg = IIneg(1:stepDots:end);

% ---- limits ----
yTop = (rho2(end)+rho2(end-1))/2;

% ---- colormap (0..1) ----
clim_plot = [0 6];
cmp = cbrewer('div','Spectral',40);
cmp = flipud(cmp);   
% alternative: cmp = turbo(256);



%%% ===================== LEFT: SOMOC =====================
ax1 = axes('Position',axpos(3,:)); hold(ax1,'on');

h = imagesc(ax1, lat_psi, rho2, r2_plot);
set(h,'AlphaData',~isnan(r2_plot));           % show only valid pixels
set(ax1,'YDir','reverse');
colormap(ax1, cmp);
clim(ax1, clim_plot);

axis(ax1, [-74 -34 35 yTop]);

% ticks/labels
xt = -75:5:-35;
xticks(ax1, xt);
lbl = arrayfun(@(x) '', xt, 'UniformOutput', false);
lbl(xt==-70) = {sprintf('70%cS',deg)};
lbl(xt==-60) = {sprintf('60%cS',deg)};
lbl(xt==-50) = {sprintf('50%cS',deg)};
lbl(xt==-40) = {sprintf('40%cS',deg)};
xticklabels(ax1, lbl);

yt = 35:0.5:37.5;
yticks(ax1, yt);
ylabel(ax1, 'Density \sigma_2 (kg/m^3)', ...
    'FontSize',fontsize,'Interpreter','tex','FontName',fontname);

% axis styling
set(ax1,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
    'TickDir','out','Box','off','Layer','top', ...
    'TickLength',[0.008667 0], ...
    'XTickLabelRotation',0, ...
    'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
ax1.Color = bg_gray;

% negative MOC dots
scatter(ax1, lat_psi(JJneg), rho2(IIneg), dotSize, 'w', 'filled', ...
    'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

% rho lines (most-frequent density of max)
plot(ax1, lat_psi, rho_SSP,      '--k', 'LineWidth', LW+0.5);
plot(ax1, lat_psi, rho_SSP_abys, '--k', 'LineWidth', LW+0.5);

% titles
% title(ax1, 'SMOC', 'FontSize',fontsize,'Interpreter','tex','FontName',fontname);

% colorbar (single, shared) — attach to left like your example
cb = colorbar(ax1,'Position',cbpos,'LineWidth',LW,'FontSize',fontsize);
cb.TickDirection = 'out';
cb.Box = 'off';
cb.FontName = fontname;
title(cb,'(Sv)','FontSize',fontsize,'FontName',fontname,'Interpreter','tex');
text(0.0, 1.1, '(B) Simulated MOC variability', 'Units','normalized','FontSize',fontsize,'Interpreter','none')

%%% ===================== RIGHT: AMOC =====================
ax2 = axes('Position',axpos(4,:)); hold(ax2,'on');

h = imagesc(ax2, lat_psi, rho2, r2_plot);
set(h,'AlphaData',~isnan(r2_plot));
set(ax2,'YDir','reverse');
colormap(ax2, cmp);
clim(ax2, clim_plot);

axis(ax2, [-34 64 35 yTop]);

% ticks/labels
xt = -35:5:75;
xticks(ax2, xt);

lbl = repmat({''}, size(xt));
for i = 1:numel(xt)
    v = xt(i);
    if v == 0
        lbl{i} = sprintf('0%c', deg);
    elseif mod(abs(v),10)==0
        if v < 0, lbl{i} = sprintf('%d%cS', abs(v), deg);
        else,     lbl{i} = sprintf('%d%cN', v, deg);
        end
    end
end
xticklabels(ax2, lbl);

yt = 35:0.5:37.5;
yticks(ax2, yt);
yticklabels(ax2, {''});

% axis styling
set(ax2,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
    'TickDir','out','Box','off','Layer','top', ...
    'TickLength',[0.004 0], ...
    'XTickLabelRotation',0, ...
    'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
ax2.Color = bg_gray;

% negative MOC dots
scatter(ax2, lat_psi(JJneg), rho2(IIneg), dotSize, 'w', 'filled', ...
    'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

% rho line
plot(ax2, lat_psi, rho_SSP, '--k', 'LineWidth', LW+0.5);

% title(ax2, 'AMOC', 'FontSize',fontsize,'Interpreter','tex','FontName',fontname);


text(ax1, 0.7, 0.99, 'SMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);

text(ax2, 0.02, 0.99, 'AMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);



% truth_std_nosmallvalue = truth_std;
% truth_std_nosmallvalue(truth_std_nosmallvalue<0.5)=nan;
% histogram(truth_std_nosmallvalue./rmse);
% median(truth_std_nosmallvalue(:)./rmse(:),'omitnan');

% exportgraphics(gcf,[model_dir_baseline '/MonteCarlo/std_moc_' cmipName '.png'],'resolution',200)

exportgraphics(gcf, fullfile(monteCarloDir, ['rmse_' cmipName '_v3.png']), 'resolution', 200)





%% Line plot
fontsize = 16;
framepos = [100 100  600   250];
figure('Position',framepos);
Pred_color =[0 0 0];

%% AMOC 26.5
Lat_plot=26.5;
rlz = 5;
framepos = [100 100  600   250];
t_range = (rlz-1)*Nsamps_r+1 : rlz*Nsamps_r;


figure('Position',framepos,'Color','w','Renderer','painters');
ax = axes; hold(ax,'on');

% ----- style -----
fontname = 'Arial';
LWmean   = LW + 1.2;
LWtrend  = LW + 0.6;
alphaBand = 0.18;
t_pred = 2015 + (1:Nsamps_r)/12;

% ----- index -----
lat_temp = Lat_plot;
lat_ind = find(abs(lat_psi-lat_temp) == min(abs(lat_psi-lat_temp)),1);

% ----- truth line -----
h2 = plot(ax, t_pred, truth_strength_mid(t_range, lat_ind), '-', 'LineWidth', LWtrend, 'Color', 'r');

y = pred_strength_mid(t_range,lat_ind);
u = uncertainty(t_range,lat_ind,pred_strength_mid_ind(lat_ind));

% ----- uncertainty band (patch first) -----
upper = y(:) + u(:);
lower = y(:) - u(:);
pband = patch(ax, [t_pred(:); flipud(t_pred(:))], [upper; flipud(lower)], ...
    Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);


uistack(pband,'bottom');  % keep under lines (still above gap is fine)

% ----- pred line -----
h1 = plot(ax, t_pred, y, '-', 'LineWidth', LWmean, 'Color', Pred_color);


% ----- axes -----
xlim(ax, [2016 2100]);
% ylim(ax, yl);
% xticks(ax, 2004:4:2024);

xlabel(ax,'Year','FontName',fontname,'FontSize',fontsize,'Interpreter','none');
ylabel(ax,'\Psi (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');

ax.FontName = fontname;
ax.FontSize = fontsize;
ax.LineWidth = LW;
ax.TickDir = 'out';
ax.Box = 'off';
ax.Layer = 'top';
ax.TickLength = [0.012 0.012];

% subtle grid (optional; comment out if you want none)
ax.YGrid = 'on';
ax.XGrid = 'off';
ax.GridAlpha = 0.12;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
% ----- title-like annotation -----
text(ax, 0.02, 0.10, sprintf('AMOC at %.1f°N', abs(Lat_plot)), ...
    'Units','normalized','FontName',fontname,'FontSize',fontsize+1, ...
    'Interpreter','none', 'Color',[0.1 0.1 0.1]);

pband.HandleVisibility = 'on';   % (optional) allow legend entry
lgd = legend(ax, [h2 h1], {'Truth','Prediction'}, ...
    'position',[0.295555562310751 0.816693331604801 0.396666659911474 0.105999997317791], 'Orientation','horizontal');
lgd.Box = 'off';
lgd.FontName = fontname;
lgd.FontSize = fontsize-1;

text(0.76, 0.9, ['$r^2= ', num2str((corr(truth_strength_mid(t_range, lat_ind),y)).^2,2) ,'$'], 'Units','normalized','FontSize',fontsize+3,'Interpreter','latex')



exportgraphics(gcf, fullfile(monteCarloDir, ['MOC_TimeSeries_' cmipName '_01.png']), 'resolution', 200)


%% AMOC 20.5S
Lat_plot = -20.5;
framepos = [100 100  600  250];
t_range = (rlz-1)*Nsamps_r+1 : rlz*Nsamps_r;


figure('Position',framepos,'Color','w','Renderer','painters');
ax = axes; hold(ax,'on');

% ----- style -----
fontname = 'Arial';
LWmean   = LW + 1.2;
LWtrend  = LW + 0.6;
alphaBand = 0.18;
t_pred = 2015 + (1:Nsamps_r)/12;

% ----- index -----
lat_temp = Lat_plot;
lat_ind = find(abs(lat_psi-lat_temp) == min(abs(lat_psi-lat_temp)),1);

% ----- truth line -----
plot(ax, t_pred, truth_strength_mid(t_range, lat_ind), '-', 'LineWidth', LWtrend, 'Color', 'r');

y = pred_strength_mid(t_range,lat_ind);
u = uncertainty(t_range,lat_ind,pred_strength_mid_ind(lat_ind));

% ----- uncertainty band (patch first) -----
upper = y(:) + u(:);
lower = y(:) - u(:);
pband = patch(ax, [t_pred(:); flipud(t_pred(:))], [upper; flipud(lower)], ...
    Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);


uistack(pband,'bottom');  % keep under lines (still above gap is fine)

% ----- pred line -----
plot(ax, t_pred, y, '-', 'LineWidth', LWmean, 'Color', Pred_color);


% ----- axes -----
xlim(ax, [2016 2100]);
% ylim(ax, yl);
% xticks(ax, 2004:4:2024);

xlabel(ax,'Year','FontName',fontname,'FontSize',fontsize,'Interpreter','none');
ylabel(ax,'\Psi (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');

ax.FontName = fontname;
ax.FontSize = fontsize;
ax.LineWidth = LW;
ax.TickDir = 'out';
ax.Box = 'off';
ax.Layer = 'top';
ax.TickLength = [0.012 0.012];

% subtle grid (optional; comment out if you want none)
ax.YGrid = 'on';
ax.XGrid = 'off';
ax.GridAlpha = 0.12;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
% ----- title-like annotation -----
text(ax, 0.02, 0.10, sprintf('AMOC at %.1f°S', abs(Lat_plot)), ...
    'Units','normalized','FontName',fontname,'FontSize',fontsize+1, ...
    'Interpreter','none', 'Color',[0.1 0.1 0.1]);
text(0.76, 0.85, ['$r^2= ', num2str((corr(truth_strength_mid(t_range, lat_ind),y)).^2,2) ,'$'], 'Units','normalized','FontSize',fontsize+3,'Interpreter','latex')

ylim([11 18])
exportgraphics(gcf, fullfile(monteCarloDir, ['MOC_TimeSeries_' cmipName '_02.png']), 'resolution', 200)



%% SMOC 50.5S
Lat_plot = -50.5;
framepos = [100 100  600  250];
t_range = (rlz-1)*Nsamps_r+1 : rlz*Nsamps_r;


figure('Position',framepos,'Color','w','Renderer','painters');
ax = axes; hold(ax,'on');

% ----- style -----
fontname = 'Arial';
LWmean   = LW + 1.2;
LWtrend  = LW + 0.6;
alphaBand = 0.18;
t_pred = 2015 + (1:Nsamps_r)/12;

% ----- index -----
lat_temp = Lat_plot;
lat_ind = find(abs(lat_psi-lat_temp) == min(abs(lat_psi-lat_temp)),1);

% ----- truth line -----
plot(ax, t_pred, truth_strength_mid(t_range, lat_ind), '-', 'LineWidth', LWtrend, 'Color', 'r');

y = pred_strength_mid(t_range,lat_ind);
u = uncertainty(t_range,lat_ind,pred_strength_mid_ind(lat_ind));

% ----- uncertainty band (patch first) -----
upper = y(:) + u(:);
lower = y(:) - u(:);
pband = patch(ax, [t_pred(:); flipud(t_pred(:))], [upper; flipud(lower)], ...
    Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);


uistack(pband,'bottom');  % keep under lines (still above gap is fine)

% ----- pred line -----
plot(ax, t_pred, y, '-', 'LineWidth', LWmean, 'Color', Pred_color);


% ----- axes -----
xlim(ax, [2016 2100]);
% ylim(ax, yl);
% xticks(ax, 2004:4:2024);

xlabel(ax,'Year','FontName',fontname,'FontSize',fontsize,'Interpreter','none');
ylabel(ax,'\Psi (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');

ax.FontName = fontname;
ax.FontSize = fontsize;
ax.LineWidth = LW;
ax.TickDir = 'out';
ax.Box = 'off';
ax.Layer = 'top';
ax.TickLength = [0.012 0.012];
ylim([15 33])
% subtle grid (optional; comment out if you want none)
ax.YGrid = 'on';
ax.XGrid = 'off';
ax.GridAlpha = 0.12;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
% ----- title-like annotation -----
text(ax, 0.02, 0.10, sprintf('Mid-depth SMOC at %.1f°S', abs(Lat_plot)), ...
    'Units','normalized','FontName',fontname,'FontSize',fontsize+1, ...
    'Interpreter','none', 'Color',[0.1 0.1 0.1]);
text(0.76, 0.9, ['$r^2= ', num2str((corr(truth_strength_mid(t_range, lat_ind),y)).^2,2) ,'$'], 'Units','normalized','FontSize',fontsize+3,'Interpreter','latex')

exportgraphics(gcf, fullfile(monteCarloDir, ['MOC_TimeSeries_' cmipName '_03.png']), 'resolution', 200)



%% abyssal SMOC 45.5S
Lat_plot = -45.5;
framepos = [100 100  600  250];
t_range = (rlz-1)*Nsamps_r+1 : rlz*Nsamps_r;


figure('Position',framepos,'Color','w','Renderer','painters');
ax = axes; hold(ax,'on');

% ----- style -----
fontname = 'Arial';
LWmean   = LW + 1.2;
LWtrend  = LW + 0.6;
alphaBand = 0.18;
alphaGap  = 0.08;

t_pred = 2015 + (1:Nsamps_r)/12;

% ----- index -----
lat_temp = Lat_plot;
lat_ind = find(abs(lat_psi-lat_temp) == min(abs(lat_psi-lat_temp)),1);

% ----- truth line -----
plot(ax, t_pred, -truth_strength_abys(t_range, lat_ind), '-', 'LineWidth', LWtrend, 'Color', 'r');

y = pred_strength_abys(t_range,lat_ind);
u = uncertainty(t_range,lat_ind,pred_strength_abys_ind(lat_ind));

% ----- uncertainty band (patch first) -----
upper = y(:) + u(:);
lower = y(:) - u(:);
pband = patch(ax, [t_pred(:); flipud(t_pred(:))], -[upper; flipud(lower)], ...
    Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);


uistack(pband,'bottom');  % keep under lines (still above gap is fine)

% ----- pred line -----
plot(ax, t_pred, -y, '-', 'LineWidth', LWmean, 'Color', Pred_color);


% ----- axes -----
xlim(ax, [2016 2100]);
% ylim(ax, yl);
% xticks(ax, 2004:4:2024);

xlabel(ax,'Year','FontName',fontname,'FontSize',fontsize,'Interpreter','none');
ylabel(ax,'-\Psi (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');

ax.FontName = fontname;
ax.FontSize = fontsize;
ax.LineWidth = LW;
ax.TickDir = 'out';
ax.Box = 'off';
ax.Layer = 'top';
ax.TickLength = [0.012 0.012];

% subtle grid (optional; comment out if you want none)
ax.YGrid = 'on';
ax.XGrid = 'off';
ax.GridAlpha = 0.12;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
% ----- title-like annotation -----
text(ax, 0.45, 0.90, sprintf('Abyssal SMOC at %.1f°S', abs(Lat_plot)), ...
    'Units','normalized','FontName',fontname,'FontSize',fontsize+1, ...
    'Interpreter','none', 'Color',[0.1 0.1 0.1]);

r2 = corr(truth_strength_abys(t_range, lat_ind), y, 'rows','complete')^2;
text(ax, 0.05, 0.90, sprintf('$r^2 = %.2f$', r2), ...
    'Units','normalized','FontSize',fontsize+3,'Interpreter','latex');
exportgraphics(gcf, fullfile(monteCarloDir, ['MOC_TimeSeries_' cmipName '_04.png']), 'resolution', 200)


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


function resolvedPath = resolveExistingMatFile(filePath, label)
resolvedPath = filePath;
if isfile(resolvedPath)
    return;
end

[~, ~, extension] = fileparts(filePath);
if isempty(extension)
    matPath = [filePath '.mat'];
    if isfile(matPath)
        resolvedPath = matPath;
        return;
    end
end

error('%s not found:\n%s', label, filePath);
end








