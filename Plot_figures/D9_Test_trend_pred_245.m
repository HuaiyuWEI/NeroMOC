clear all
close all
clc
fontsize = 15; 
LW = 1;

addpath(genpath('D:\OneDrive - University of California\MATLAB toolboxs'))
addpath(genpath('D:\OneDrive - University of California\MATLAB Codes\CMIP6'))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
covariate_name = 'obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5';%obp_mascon

NNname = 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';

cmipName = 'SSP245'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load baseline model
basedir = 'E:\Analysis2026\ACCESS_hist+SSP585\ASMOC\results_LPF2Year';
model_dir_baseline = fullfile(basedir, NNname, covariate_name,'MonteCarlo');
cd(model_dir_baseline)

load(['R2_' cmipName '_PNoise40_SSHNoise1_MC500.mat'])

Nrho = 18;
Nlat = 140 ;
truth =reshape(Psi,[length(Psi)/5,5,Nlat,Nrho]);

pred_SSP =reshape(pred_mean_yz,[length(pred_mean_yz)/5,5,Nrho,Nlat]);



Nsamps_rlz = length(truth);
Lat_26_ind= find(abs(lat_psi-26.5) == min(abs(lat_psi-26.5)));
lat_psi(Lat_26_ind)
figure
hold on
for rlz = 1:5
plot(squeeze(truth(:,rlz,Lat_26_ind,10)))
end

load('../TestR2_SSP245.mat', 'rho2')
% rho2 = rho2-1000;


t_access = 2015+[1:Nsamps_rlz]/12;
%% trend in ACCESS (SSP245; truth)

psi_yz = squeeze( mean(truth , 2));

t_1 = 1;
t_2 = Nsamps_rlz;

trend_psi = nan*squeeze(mean(psi_yz,1));
trend_rval = nan*squeeze(mean(psi_yz,1));
trend_pval = nan*squeeze(mean(psi_yz,1));
for k = 1: Nlat
    for j = 1: Nrho
coeff = polyfit(t_access(t_1:t_2),squeeze(psi_yz(t_1:t_2,k,j)),1);
trend_psi(k,j) = coeff(1);
temp = coeff(2)+coeff(1)*t_access(t_1:t_2);

[r,~] = corr(temp',squeeze(psi_yz(t_1:t_2,k,j)));
trend_rval(k,j) = r;
DOF = Nsamps_rlz/10;
tvals = r .* sqrt(DOF ./ (1 - r.^2));
trend_pval(k,j) = 2 * (1 - tcdf(abs(tvals), DOF)); 

    end
end

%% find density level at which MOC maxima occur most frequently
%%% Time mean value of diagnosed MOC; and define MOC strength
temp = 'E:\Analysis2026\ACCESS_hist+SSP585\ASMOC\results_LPF2Year\FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year\obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5';
y = load([temp '/Pred_' cmipName]).y;

Nsamps = length(y)
MOC= reshape(y,[Nsamps,Nlat,Nrho]);
MOC_mean = squeeze(mean(MOC))';

rho2 = rho2-1000;
SmoothParam = 5;
SmoothMethod = 'moving';


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

% %%% MOC strength
% truth_strength_mid = zeros(Nsamps,Nlat);
% truth_strength_abys = zeros(Nsamps,Nlat);
% pred_strength_mid = zeros(Nsamps,Nlat);
% pred_strength_abys = zeros(Nsamps,Nlat);
% for jj = 1:Nlat
% truth_strength_mid(:,jj) = truth(:,jj,rho_ind_mean(jj));
% truth_strength_abys (:,jj) = truth(:,jj,rho_ind_abys_mean(jj));
% pred_strength_mid(:,jj) = pred(:,jj,rho_ind_mean(jj));
% pred_strength_abys (:,jj) = pred(:,jj,rho_ind_abys_mean(jj));
% end
% pred_strength_mid_ind = rho_ind_mean;
% pred_strength_abys_ind = rho_ind_abys_mean;


%%




%% plot the root mean squared error of the prediction
%%% and %% plot the std of the moc (truth)
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

% ---- plot field ----
r2_plot = trend_psi';
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
clim_plot = [-0.32 0.32];
cmp = cbrewer('div','RdBu',40);
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
title(cb,'(Sv yr^{-1})','FontSize',fontsize,'FontName',fontname,'Interpreter','tex');
text(0.0, 1.1, '(A) Simulated trend', 'Units','normalized','FontSize',fontsize,'Interpreter','none')


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






%% trend in ACCESS (SSP245; pred)

psi_yz = squeeze( mean(pred_SSP , 2));
psi_yz = permute(psi_yz,[1 3 2]);

t_1 = 1;
t_2 = Nsamps_rlz;

trend_psi = nan*squeeze(mean(psi_yz,1));
trend_rval = nan*squeeze(mean(psi_yz,1));
trend_pval = nan*squeeze(mean(psi_yz,1));
for k = 1: Nlat
    for j = 1: Nrho
coeff = polyfit(t_access(t_1:t_2),squeeze(psi_yz(t_1:t_2,k,j)),1);
trend_psi(k,j) = coeff(1);
temp = coeff(2)+coeff(1)*t_access(t_1:t_2);

[r,~] = corr(temp',squeeze(psi_yz(t_1:t_2,k,j)));
trend_rval(k,j) = r;

DOF = Nsamps_rlz/10;
tvals = r .* sqrt(DOF ./ (1 - r.^2));
trend_pval(k,j) = 2 * (1 - tcdf(abs(tvals), DOF)); 



    end
end


[~,rho_ind] =max(psi_yz,[],3);

psi_yz_neg = psi_yz;
psi_yz_neg(psi_yz_neg>0) = nan;
[~,rho_ind_abys] = min(psi_yz_neg,[],3);

rho_ind = squeeze(rho_ind);
rho_ind_abys = squeeze(rho_ind_abys);

rho_ind_alltime= mode(rho_ind,1);
% rho_ind_alltime(lat_psi < -55.5) = nan;
% figure
% plot(rho_ind_alltime);hold on
rho_ind_alltime = smooth(lat_psi,rho_ind_alltime,SmoothParam,SmoothMethod)
rho_ind_alltime = round(rho_ind_alltime);
% plot(rho_ind_alltime);

rho_max = rho2(rho_ind_alltime);
rho_max(lat_psi < -55.5) = nan;

rho_ind_abys_alltime= mode(rho_ind_abys,1);
rho_max_abys = rho2(rho_ind_abys_alltime);

rho_ind_abys_alltime = smooth(lat_psi,rho_ind_abys_alltime,SmoothParam,SmoothMethod)
rho_ind_abys_alltime = round(rho_ind_abys_alltime);



%%



%%%%%%%%%%% STD
% ---- plot field ----
r2_plot = trend_psi';
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
clim_plot = [-0.3 0.3];
cmp = cbrewer('div','RdBu',40);
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


title(cb,'(Sv yr^{-1})','FontSize',fontsize,'FontName',fontname,'Interpreter','tex');
text(0.0, 1.1, '(B) Reconstructed trend', 'Units','normalized','FontSize',fontsize,'Interpreter','none')

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





exportgraphics(gcf,[model_dir_baseline '\' cmipName  '_trend_v3.png'],'resolution',200)



cd(model_dir_baseline)







