%%% Define MOC strength as the MOC variability at density level where
%%% timemean MOC is maximized

clear all
close all
clc


addpath(genpath('D:\OneDrive - University of California\MATLAB toolboxs'))
addpath(genpath('D:\OneDrive - University of California\MATLAB Codes\CMIP6'))

clim_minmax = [-0.2 0.2];
basedir = 'E:\Analysis2026\ACCESS_hist+SSP585\ASMOC\results_LPF2Year';

SmoothParam = 5;
SmoothMethod = 'moving';

y_ticks = [35:0.5:38];

%%  load baseline model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cmipName = 'SSP126';
covariate_name = 'obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5'
NNname = 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_dir_baseline = fullfile(basedir,NNname,covariate_name);
cd(model_dir_baseline)


%%% use MonteCarlo Results to compute correlation
cd("MonteCarlo")
load(['R2_' cmipName '_PNoise40_SSHNoise1_MC500.mat'])
[Nsamps,Nrho,Nlat]= size(pred_mean_yz);
truth = reshape(Psi,[length(Psi),140,18]); 
pred = permute(pred_mean_yz, [1,3,2]);
rmse = squeeze(sqrt(mean((truth - pred).^2)))';
save(['rmse_' cmipName '_PNoise40_SSHNoise1_MC500.mat'],"rmse")
truth_std = squeeze(std(truth,1))';

uncertainty = sqrt(permute(pred_variance_yz, [1,3,2]));
pred_upper = pred + uncertainty;
pred_lower = pred - uncertainty;
r2_baseline = nan(Nrho,Nlat);
for j = 1:Nlat
    for k = 1:Nrho
        r2_baseline(k,j) = (corr(truth(:,j,k),pred(:,j,k))).^2;
    end
end



%%%%%%%%% r2 for detrended reconstruction and diagnostic
N_realization = 5;
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
temp = 'E:\Analysis2026\ACCESS_hist+SSP585\ASMOC\results_LPF2Year\FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year\obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5';
y = load([temp '/Pred_' cmipName]).y;
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
[test,temp] = max(MOC_mean,[],'all');
[idx_rho, temp2] = ind2sub(size(MOC_mean), temp);
Lax_maxSOMOC_mid = lat_psi(temp2)

[test,temp] = min(MOC_mean,[],'all');
[idx_rho, temp2] = ind2sub(size(MOC_mean), temp);
Lax_maxSOMOC_abys = lat_psi(temp2)

AMOC_mean=MOC_mean;
AMOC_mean(:,1:41)=nan;
[test,temp] = max(AMOC_mean,[],'all');
[idx_rho, temp2] = ind2sub(size(AMOC_mean), temp);
Lax_maxAMOC = lat_psi(temp2)


%%

framepos = [100 50 1300 600];
fh= figure('Position',framepos);


% ---- layout ----
axpos(1,:) = [0.08 0.55 0.15 0.32];
axpos(2,:) = [0.24 0.55 0.35 0.32];
axpos(3,:) = [0.08 0.10 0.15 0.32];
axpos(4,:) = [0.24 0.10 0.35 0.32];
cbpos      = [0.6 0.1 0.01 0.77];
% ---- style ----
fontsize = 12;
LW = 1;
fontname = 'Arial';
deg = char(176);

bg_gray = [0.90 0.90 0.90];     % NaN/background
dotSize = 8;


%% plot the reconstruction skill of the first senario
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
cmp = pmkmp(40,'Swtth');  
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
title(cb,'r^2','FontSize',fontsize,'FontName',fontname,'Interpreter','tex');
text(0.0, 1.1, '(A) SSP1-2.6', 'Units','normalized','FontSize',fontsize,'Interpreter','none')


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





%% load the reconstruction skill of the second senario




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cmipName = 'SSP370';
covariate_name = 'obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5'
% NNname = 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_dir_baseline = fullfile(basedir,NNname,covariate_name);
cd(model_dir_baseline)


%%% use MonteCarlo Results to compute correlation
cd("MonteCarlo")
load(['R2_' cmipName '_PNoise40_SSHNoise1_MC500.mat'])
[Nsamps,Nrho,Nlat]= size(pred_mean_yz);
truth = reshape(Psi,[length(Psi),140,18]); 
pred = permute(pred_mean_yz, [1,3,2]);
rmse = squeeze(sqrt(mean((truth - pred).^2)))';
save(['rmse_' cmipName '_PNoise40_SSHNoise1_MC500.mat'],"rmse")
truth_std = squeeze(std(truth,1))';

uncertainty = sqrt(permute(pred_variance_yz, [1,3,2]));
pred_upper = pred + uncertainty;
pred_lower = pred - uncertainty;
r2_baseline = nan(Nrho,Nlat);
for j = 1:Nlat
    for k = 1:Nrho
        r2_baseline(k,j) = (corr(truth(:,j,k),pred(:,j,k))).^2;
    end
end



%%%%%%%%% r2 for detrended reconstruction and diagnostic
N_realization = 5;
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
temp = 'E:\Analysis2026\ACCESS_hist+SSP585\ASMOC\results_LPF2Year\FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year\obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5';
y = load([temp '/Pred_' cmipName]).y;
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
[test,temp] = max(MOC_mean,[],'all');
[idx_rho, temp2] = ind2sub(size(MOC_mean), temp);
Lax_maxSOMOC_mid = lat_psi(temp2)

[test,temp] = min(MOC_mean,[],'all');
[idx_rho, temp2] = ind2sub(size(MOC_mean), temp);
Lax_maxSOMOC_abys = lat_psi(temp2)

AMOC_mean=MOC_mean;
AMOC_mean(:,1:41)=nan;
[test,temp] = max(AMOC_mean,[],'all');
[idx_rho, temp2] = ind2sub(size(AMOC_mean), temp);
Lax_maxAMOC = lat_psi(temp2)





%% plot the reconstruction skill of the second senario
% ---- plot field ----
r2_plot = r2_baseline;
r2_plot(MOC_std < 0.5) = nan;

% ---- masks (compute once) ----
MaskNeg = (MOC_mean < 0) & (MOC_std > 0.5);  % [lat x rho] or [rho x lat] depending on your arrays
[JJneg, IIneg] = find(MaskNeg');              % JJneg->lat index, IIneg->rho index (match your earlier convention)

stepDots = 1;                                  % 1=all, 2=sparser
JJneg = JJneg(1:stepDots:end);
IIneg = IIneg(1:stepDots:end);

% ---- limits ----
yTop = (rho2(end)+rho2(end-1))/2;

% ---- colormap (0..1) ----
clim_plot = [0 1];
cmp = pmkmp(40,'Swtth');  
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



text(0.0, 1.1, '(B) SSP3-7.0', 'Units','normalized','FontSize',fontsize,'Interpreter','none')

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


text(ax1, 0.02, 0.98, 'SMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);

text(ax2, 0.02, 0.98, 'AMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);





exportgraphics(gcf,[model_dir_baseline '/MonteCarlo/SSP126_SSP370_r2_v3.png'],'resolution',200)

