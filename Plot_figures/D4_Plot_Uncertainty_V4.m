
clear all
close all
clc
% fontsize = 18; LW = 2;
fontsize = 16;
LW = 1;
framepos = [100 100  1800   600];
axpos = zeros(5,4);
axpos(1,:) = [0.08 0.6 0.4 0.34];
axpos(2,:) = [0.51 0.6 0.4 0.34];


axpos(3,:) = [0.08 0.08 0.115 0.4];
axpos(4,:) = [0.2 0.08 0.275 0.4];


axpos(5,:) = [0.08+0.43 0.08 0.115 0.4];
axpos(6,:) = [0.2+0.43 0.08 0.275 0.4];


addpath(genpath('D:\OneDrive - University of California\MATLAB toolboxs'))
addpath(genpath('D:\OneDrive - University of California\MATLAB Codes\CMIP6'))

clim_minmax = [-0.2 0.2];
basedir = 'E:\Analysis2026\ACCESS_hist+SSP585\ASMOC\results_LPF2Year';

SmoothParam = 5;
SmoothMethod = 'moving';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cmipName = 'SSP245';
NNname = 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
rlz = 5;








fh= figure('Position',framepos);
%% OBP+SSH+ZWS line plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
covariate_name = 'obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5'
model_dir_baseline = fullfile(basedir,NNname,covariate_name);
cd(model_dir_baseline)

%%% use MonteCarlo Results to compute correlation
cd("MonteCarlo")
load(['R2_' cmipName '_PNoise40_SSHNoise1_MC500.mat'])
[Nsamps,Nrho,Nlat]= size(pred_mean_yz);
truth = reshape(Psi,[length(Psi),140,18]); 
pred = permute(pred_mean_yz, [1,3,2]);
rmse = squeeze(sqrt(mean((truth - pred).^2)))';
truth_std = squeeze(std(truth,1))';
uncertainty = sqrt(permute(pred_variance_yz, [1,3,2]));
pred_upper = pred + uncertainty;
pred_lower = pred - uncertainty;

N_realization = 5;
Nsamps_r = Nsamps/N_realization;


%%% Time mean value of diagnosed MOC; and define MOC strength
temp = 'E:\Analysis2026\ACCESS_hist+SSP585\ASMOC\results_LPF2Year\FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year\obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5';
y = load([temp '/Pred_' cmipName]).y;
Nsamps = length(y);
MOC= reshape(y,[Nsamps,Nlat,Nrho]);
MOC_mean = squeeze(mean(MOC))';

rho2 = rho2-1000;

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
truth_strength_mid = zeros(Nsamps,Nlat);
truth_strength_abys = zeros(Nsamps,Nlat);
pred_strength_mid = zeros(Nsamps,Nlat);
pred_strength_abys = zeros(Nsamps,Nlat);
for jj = 1:Nlat
truth_strength_mid(:,jj) = truth(:,jj,rho_ind_mean(jj));
truth_strength_abys (:,jj) = truth(:,jj,rho_ind_abys_mean(jj));
pred_strength_mid(:,jj) = pred(:,jj,rho_ind_mean(jj));
pred_strength_abys (:,jj) = pred(:,jj,rho_ind_abys_mean(jj));
end
pred_strength_mid_ind = rho_ind_mean;
pred_strength_abys_ind = rho_ind_abys_mean;



%% Line plot

hold on

Pred_color =[0 0 0];

%% AMOC 26.5
Lat_plot=26.5;
rlz = 5;

t_range = (rlz-1)*Nsamps_r+1 : rlz*Nsamps_r;



ax = subplot('Position',axpos(1,:)); hold(ax,'on');

% ----- style -----
fontname = 'Arial';
LWmean   = LW + 1.2;
LWtrend  = LW + 0.6;
alphaBand = 0.18;
alphaGap  = 0.08;

t_pred = 2015+[1:Nsamps_r]/12;

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
ylim(ax, [10 18]);
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
text(ax, 0.02, 0.10, 'Inputs: $p_b$, $\eta$, and $u^\mathrm{wind}$', ...
    'Units','normalized','FontName',fontname,'FontSize',fontsize+5, ...
    'Interpreter','latex', 'Color',[0.1 0.1 0.1]);
pband.HandleVisibility = 'on';   % (optional) allow legend entry
lgd = legend(ax, [h2 h1 pband], {'Truth','Prediction','Uncertainty'}, ...
    'Location','northeast', 'Orientation','horizontal');
lgd.Box = 'off';
lgd.FontName = fontname;
lgd.FontSize = fontsize-1;

% text(0.76, 0.9, ['$r^2= ', num2str((corr(truth_strength_mid(t_range, lat_ind),y)).^2,2) ,'$'], 'Units','normalized','FontSize',fontsize+3,'Interpreter','latex')


text(0.0, 1.06, 'A', 'Units','normalized', ...
    'FontSize',fontsize, 'FontWeight','bold', ...
    'Interpreter','none')


% text(0.02, 0.1, '(a)', 'Units','normalized','FontSize',fontsize,'Interpreter','latex')

    %% OBP+SSH+ZWS spatial pattern of uncertainty

uncertainty_mean = squeeze(mean(uncertainty))';

if(rho2(1)>1000)
rho2 = rho2-1000;
end
cbpos      = [0.6 0.1 0.01 0.77];

% ---- style ----
LW = 1;
fontname = 'Arial';
deg = char(176);

bg_gray = [0.90 0.90 0.90];     % NaN/background
dotSize = 8;

% ---- plot field ----
r2_plot = uncertainty_mean;


% ---- masks (compute once) ----
MaskNeg = (MOC_mean < 0);   % [lat x rho] or [rho x lat] depending on your arrays
[JJneg, IIneg] = find(MaskNeg');              % JJneg->lat index, IIneg->rho index (match your earlier convention)

stepDots = 1;                                  % 1=all, 2=sparser
JJneg = JJneg(1:stepDots:end);
IIneg = IIneg(1:stepDots:end);

% ---- limits ----
yTop = (rho2(end)+rho2(end-1))/2;

% ---- colormap (0..1) ----
clim_plot = [0 3];
cmp = cbrewer('div','Spectral',40)
cmp = flipud(cmp);   


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

text(0.0, 1.06, 'C', 'Units','normalized', ...
    'FontSize',fontsize, 'FontWeight','bold', ...
    'Interpreter','none')

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


text(ax1, 0.68, 0.99, 'SMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);

text(ax2, 0.02, 0.99, 'AMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);



%% OBP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
covariate_name = 'obp_mascon_V5'
model_dir_baseline = fullfile(basedir,NNname,covariate_name);
cd(model_dir_baseline)

%%% use MonteCarlo Results to compute correlation
cd("MonteCarlo")
load(['R2_' cmipName '_PNoise40_SSHNoise1_MC500.mat'])
[Nsamps,Nrho,Nlat]= size(pred_mean_yz);
truth = reshape(Psi,[length(Psi),140,18]); 
pred = permute(pred_mean_yz, [1,3,2]);
rmse = squeeze(sqrt(mean((truth - pred).^2)))';
truth_std = squeeze(std(truth,1))';
uncertainty = sqrt(permute(pred_variance_yz, [1,3,2]));
pred_upper = pred + uncertainty;
pred_lower = pred - uncertainty;

N_realization = 5;
Nsamps_r = Nsamps/N_realization;


%%% Time mean value of diagnosed MOC; and define MOC strength
% temp = 'E:\Analysis2026\ACCESS_hist+SSP585\ASMOC\results_LPF2Year\FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year\obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5';
y = load([temp '/Pred_' cmipName]).y;
Nsamps = length(y);
MOC= reshape(y,[Nsamps,Nlat,Nrho]);
MOC_mean = squeeze(mean(MOC))';
rho2 = rho2-1000;
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
truth_strength_mid = zeros(Nsamps,Nlat);
truth_strength_abys = zeros(Nsamps,Nlat);
pred_strength_mid = zeros(Nsamps,Nlat);
pred_strength_abys = zeros(Nsamps,Nlat);
for jj = 1:Nlat
truth_strength_mid(:,jj) = truth(:,jj,rho_ind_mean(jj));
truth_strength_abys (:,jj) = truth(:,jj,rho_ind_abys_mean(jj));
pred_strength_mid(:,jj) = pred(:,jj,rho_ind_mean(jj));
pred_strength_abys (:,jj) = pred(:,jj,rho_ind_abys_mean(jj));
end
pred_strength_mid_ind = rho_ind_mean;
pred_strength_abys_ind = rho_ind_abys_mean;




ax = subplot('Position',axpos(2,:)); hold(ax,'on');

% ----- style -----
fontname = 'Arial';
LWmean   = LW + 1.2;
LWtrend  = LW + 0.6;
alphaBand = 0.18;
alphaGap  = 0.08;

t_pred = 2015+[1:Nsamps_r]/12;

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
ylim(ax, [10 18]);
% xticks(ax, 2004:4:2024);

xlabel(ax,'Year','FontName',fontname,'FontSize',fontsize,'Interpreter','none');
% ylabel(ax,'\Psi (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');
yticklabels('')

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

text(ax, 0.02, 0.10, 'Input: $p_b$', ...
    'Units','normalized','FontName',fontname,'FontSize',fontsize+5, ...
    'Interpreter','latex', 'Color',[0.1 0.1 0.1]);



text(0.0, 1.06, 'B', 'Units','normalized', ...
    'FontSize',fontsize, 'FontWeight','bold', ...
    'Interpreter','none')




%% OBP spatial pattern of uncertainty

uncertainty_mean = squeeze(mean(uncertainty))';

if(rho2(1)>1000)
rho2 = rho2-1000;
end
cbpos      = [0.91 0.08 0.007 0.396];

% ---- style ----

LW = 1;
fontname = 'Arial';
deg = char(176);

bg_gray = [0.90 0.90 0.90];     % NaN/background
dotSize = 8;

% ---- plot field ----
r2_plot = uncertainty_mean;


% ---- masks (compute once) ----
MaskNeg = (MOC_mean < 0);   % [lat x rho] or [rho x lat] depending on your arrays
[JJneg, IIneg] = find(MaskNeg');              % JJneg->lat index, IIneg->rho index (match your earlier convention)

stepDots = 1;                                  % 1=all, 2=sparser
JJneg = JJneg(1:stepDots:end);
IIneg = IIneg(1:stepDots:end);

% ---- limits ----
yTop = (rho2(end)+rho2(end-1))/2;

% ---- colormap (0..1) ----
clim_plot = [0 3];
cmp = cbrewer('div','Spectral',40)
cmp = flipud(cmp);   


%%% ===================== LEFT: SOMOC =====================
ax1 = axes('Position',axpos(5,:)); hold(ax1,'on');

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
% ylabel(ax1, 'Density \sigma_2 (kg/m^3)', ...
%     'FontSize',fontsize,'Interpreter','tex','FontName',fontname);
yticklabels('')
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



cb = colorbar(ax1,'Position',cbpos,'LineWidth',LW,'FontSize',fontsize);
cb.TickDirection = 'out';
cb.Box = 'off';
cb.FontName = fontname;
title(cb,'(Sv)','FontSize',fontsize,'FontName',fontname,'Interpreter','tex');

text(0.0, 1.06, 'D', 'Units','normalized', ...
    'FontSize',fontsize, 'FontWeight','bold', ...
    'Interpreter','none')

%%% ===================== RIGHT: AMOC =====================
ax2 = axes('Position',axpos(6,:)); hold(ax2,'on');

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


text(ax1, 0.68, 0.99, 'SMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);

text(ax2, 0.02, 0.99, 'AMOC', 'Units','normalized', ...
    'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    'Color',[0.1 0.1 0.1]);




%%


exportgraphics(gcf,fullfile(basedir,NNname,['/AMOC_26_' cmipName '_v3.png']),'resolution',200)


cd(fullfile(basedir,NNname))


