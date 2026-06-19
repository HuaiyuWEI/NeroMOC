clear all
close all
clc

CT= cbrewer('qual','Set1',8);
SmoothParam = 5;
SmoothMethod = 'moving';
addpath(genpath('D:\OneDrive - University of California\MATLAB toolboxs'))
addpath(genpath('D:\OneDrive - University of California\MATLAB Codes\CMIP6'))

Pred_color = [0 0 0];
Pred_color02 = CT(2,:);



%%%% remove the first and last parts of the time series to avoid filter edge effect
edge = 12;   % change this number as needed


%% Load DBNN prediction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
covariate_name = 'obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5';%+ssh_mascon_V5+uas_mascon_V5
NNname = 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year'; %
basedir = 'E:\Analysis2026\ACCESS_hist+SSP585\ASMOC\results_LPF2Year';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_dir_baseline = fullfile(basedir, NNname, covariate_name);
cd(model_dir_baseline)
cd('./RealWorld')
filename = 'Pred_RealWorld'; %
load(filename)
%%%% remove the first and last parts of the time series to avoid filter edge effect
pred_yz = pred_yz(edge+1:end-edge,:,:);
pred_yz_std = pred_yz_std(edge+1:end-edge,:,:);
rho2 = rho2 - 1000;
[Nsamps, Nrho, Nlat0] = size(pred_yz);
lat_ACCESS = lat;
clear lat

t_pred = 2002+4/12+1+[0:Nsamps-1]/12;
t_range = [t_pred(1) t_pred(end)];
 

%%% uncertainty based on SSP245 test
load('../MonteCarlo/rmse_SSP245_PNoise40_SSHNoise1_MC500.mat')
NN_rmse = rmse;
%%% epistemic uncertainty
NN_epist = pred_yz_std;
%%% total uncertainty
NN_rmse = repmat(reshape(NN_rmse,[1,Nrho,Nlat0]),[Nsamps,1,1])+NN_epist;

%%% define MOC strength
pred_yz_mean = squeeze(mean(pred_yz));

% mid-depth cell
[~,rho_ind_mean] = max(pred_yz_mean,[],1);
rho_ind_mean = smooth(lat_ACCESS,rho_ind_mean,SmoothParam,SmoothMethod);
rho_ind_mean = round(rho_ind_mean);
rho_DBNN = rho2(rho_ind_mean);
rho_DBNN(lat_ACCESS < -55.5) = nan;

% abyssal cell
[~,rho_ind_abys_mean] = min(pred_yz_mean,[],1);
rho_ind_abys_mean = smooth(lat_ACCESS,rho_ind_abys_mean,SmoothParam,SmoothMethod);
rho_ind_abys_mean = round(rho_ind_abys_mean);
rho_DBNN_abys = rho2(rho_ind_abys_mean);


%%% MOC strength
pred_strength_mid = zeros(Nsamps,Nlat0);
pred_strength_abys = zeros(Nsamps,Nlat0);
uncertainty_mid = zeros(Nsamps,Nlat0);
uncertainty_abys = zeros(Nsamps,Nlat0);
for jj = 1:Nlat0
pred_strength_mid(:,jj) = pred_yz(:,rho_ind_mean(jj),jj);
pred_strength_abys(:,jj) = pred_yz(:,rho_ind_abys_mean(jj),jj);
uncertainty_mid(:,jj) = NN_rmse(:,rho_ind_mean(jj),jj);
uncertainty_abys(:,jj) = NN_rmse(:,rho_ind_abys_mean(jj),jj);
end


DBNN_rho_ind_mid = rho_ind_mean;
DBNN_rho_ind_abys = rho_ind_abys_mean;





%% Rapid
load('E:\Data_RAPID\Rapid_LPF.mat')
%%%% remove the first and last 24 months to avoid filter edge effect
RAPID_monthly_LPF = RAPID_monthly_LPF(edge+1:end-edge);
t_RAPID = t_year(edge+1:end-edge);
%%%


%% ECCO
%%% V4r3
MOC_ecco_LPF = load('E:\Data_ECCO\ECCOV4r3\PSI_LPF.mat').MOC_ecco_LPF;
t_ecco = load('E:\Data_ECCO\ECCOV4r3\PSI_LPF.mat').t_year_ecco;
lat = load('E:\Data_ECCO\ECCOV4r3\myproducts_monthly\PSI.mat').lat;
MOC_ecco_LPF(:,1:42,:) = 0; %ignore sigma2<35

MOC_ecco_LPF = MOC_ecco_LPF-MOC_ecco_LPF(:,end,:); % [time,rho,lat]


lat_ecco = 0.5*( lat(1:end-1)+ lat(2:end));
PSI_ECCO = 0.5* (MOC_ecco_LPF(edge+1:end-edge,:,1:end-1)+MOC_ecco_LPF(edge+1:end-edge,:,2:end));
t_ecco = t_ecco(edge+1:end-edge);
% MOC strength
PSI_ECCO_mean = squeeze(mean(PSI_ECCO,1));
[~,rho_ind_mean] = max(PSI_ECCO_mean,[],1);
rho_ind_mean = smooth(lat_ecco,rho_ind_mean,SmoothParam,SmoothMethod);
rho_ind_mean = round(rho_ind_mean);
ECCO_strength_AMOC = zeros(length(PSI_ECCO(:,1,1)),length(PSI_ECCO(1,1,:)));
for jj = 1:length(PSI_ECCO(1,1,:))
ECCO_strength_AMOC(:,jj) = PSI_ECCO(:,rho_ind_mean(jj),jj);
end
clear rho_ind_mean




MOC_ecco_LPF = load('E:\Data_ECCO\ECCOV4r3\PSI_SOMOC_LPF.mat').MOC_ecco_LPF;
t_ecco = load('E:\Data_ECCO\ECCOV4r3\PSI_SOMOC_LPF.mat').t_year_ecco;
lat = load('E:\Data_ECCO\ECCOV4r3\myproducts_monthly\PSI.mat').lat;
MOC_ecco_LPF(:,1:42,:) = 0; %ignore sigma2<35

MOC_ecco_LPF = MOC_ecco_LPF-MOC_ecco_LPF(:,end,:); % [time,rho,lat]


lat_ecco = 0.5*( lat(1:end-1)+ lat(2:end));
PSI_ECCO = 0.5* (MOC_ecco_LPF(edge+1:end-edge,:,1:end-1)+MOC_ecco_LPF(edge+1:end-edge,:,2:end));
t_ecco = t_ecco(edge+1:end-edge);
% MOC strength
PSI_ECCO_mean = squeeze(mean(PSI_ECCO,1));
[~,rho_ind_mean] = max(PSI_ECCO_mean,[],1);
rho_ind_mean = smooth(lat_ecco,rho_ind_mean,SmoothParam,SmoothMethod);
rho_ind_mean = round(rho_ind_mean);
ECCO_strength_SOMOC_mid = zeros(length(PSI_ECCO(:,1,1)),length(PSI_ECCO(1,1,:)));
for jj = 1:length(PSI_ECCO(1,1,:))
ECCO_strength_SOMOC_mid(:,jj) = PSI_ECCO(:,rho_ind_mean(jj),jj);
end
clear rho_ind_mean


[~,rho_ind_mean] = min(PSI_ECCO_mean,[],1);
rho_ind_mean = smooth(lat_ecco,rho_ind_mean,SmoothParam,SmoothMethod);
rho_ind_mean = round(rho_ind_mean);
ECCO_strength_SOMOC_abys = zeros(length(PSI_ECCO(:,1,1)),length(PSI_ECCO(1,1,:)));
for jj = 1:length(PSI_ECCO(1,1,:))
ECCO_strength_SOMOC_abys(:,jj) = PSI_ECCO(:,rho_ind_mean(jj),jj);
end
clear rho_ind_mean

rho_ecco = load('E:\Data_ECCO\ECCOV4r3\myproducts_monthly\PSI.mat').dens_bnds;




% 
% figure
% % pcolor(lat_ecco,rho_ecco,PSI_ECCO_mean)
% % pcolor(lat,rho_ecco,squeeze(mean(PSItot_AMOC,3))'/1e6)
% % pcolor(lat,rho_ecco,squeeze(mean(PSItot,3))'/1e6)
% pcolor(lat,rho_ecco,squeeze(mean(PSI,3))'/1e6)
% shading flat
% colormap
% box on 
% grid on
% set(gca,'YDir','reverse');
% set(gca,'Layer','top','tickLabelinterpreter', 'latex')
% clim([-20 20])
% ylim([1035 1038])
% cmocean('red',80)




%% compute trend 

significant = 0.1;

nMC = 500;  % number of perturbations
[nt, nk, nj] = size(pred_yz);

trend_mean  = nan(nk,nj);
trend_ci95  = nan(2,nk,nj); % slope confidence interval
trend_pval  = nan(nk,nj);   % slope p-value from MC
intercepts_mean = nan(nk,nj);

corr_mean   = nan(nk,nj);
corr_ci95   = nan(2,nk,nj); % correlation confidence interval
corr_pval   = nan(nk,nj);   % correlation significance

DOF_yz   = nan(nk,nj); 
for k = 1:nk
    k
    for j = 1:nj
        y = squeeze(pred_yz(:,k,j)); % time series
        sigma = NN_rmse(:,k,j);

        slopes = nan(nMC,1);
        rvals  = nan(nMC,1);
        pvals  = nan(nMC,1);


        % %%% this doesn't work as a perfect linear trend is autocorrelated
        % %%% with itself, no matter how long the lag is.
        % %%% Compute effective DOF for X and Y
        % % fit linear trend
        % coeff = polyfit(t_pred, y, 1);
        % y_fit = polyval(coeff, t_pred);  
        % lag_max = 1*12;
        % Yautocov = zeros(1,lag_max+1);
        % Lautocov = zeros(1,lag_max+1);
        % for lag = 1:length(Lautocov)      
        %   Lautocov(lag) = lag-1;
        %   Yautocov(lag) = corr(y(1:end-lag+1),y(lag:end));
        % end
        % Te_Y = Lautocov(find(Yautocov<1/exp(1),1,'first'));
        % if (isempty(Te_Y))
        %   Te_Y = lag_max;
        %   disp(['Covariance limit reached']);
        % end
        % 
        % DOF = length(y)/Te_Y;
        % DOF_yz(k,j) = DOF;

        DOF = 25;%Nsamps/10;
        % add noise; monta carlo
        for imc = 1:nMC
            % add Gaussian noise with std = RMSE
            y_mc = y + sigma.*randn(nt,1);
            % fit linear trend
            coeff = polyfit(t_pred, y_mc, 1);
            slopes(imc) = coeff(1);
            intercepts(imc) = coeff(2);
            % fitted line
            y_fit = polyval(coeff, t_pred);
            % correlation between fitted line and noisy data
            [r,~] = corr(y_fit', y_mc);
            rvals(imc) = r;
            tvals = r .* sqrt(DOF ./ (1 - r.^2));
            pvals(imc) = 2 * (1 - tcdf(abs(tvals), DOF)); 
        end

        % summarize slope distribution
        trend_mean(k,j)   = mean(slopes);
        trend_ci95(:,k,j) = prctile(slopes,[2.5 97.5]);
        trend_pval(k,j)   = 2*min(mean(slopes>0), mean(slopes<0));
        intercepts_mean(k,j)   = mean(intercepts);
        % summarize correlation distribution
        corr_mean(k,j)   = mean(rvals);
        corr_ci95(:,k,j) = prctile(rvals,[2.5 97.5]);
        corr_pval(k,j)   = mean(pvals < significant); % fraction of runs significant
    end
end




%% line plot - AMOC (multi-lat)  [final, cleaned]

fontsize = 15;
LW = 1;
MS = 10;  %#ok<NASGU>  % (kept in case you later use markers)

% ---- figure / layout ----
figH = figure(1);
set(figH,'Position',[100 100 1600 600],'Color','w','Renderer','painters');
tlo = tiledlayout(figH, 3, 3, 'TileSpacing','compact', 'Padding','compact');

% ---- style ----
fontname  = 'Arial';
LWmean    = LW + 1.2;
LWtrend   = LW + 0.6;
alphaBand = 0.18;
alphaGap  = 0.08;

Pred_color       = [0 0 0];           % DBNN mean
trend_up_color   = [0.85 0.2 0.2];    % increasing = red
trend_down_color = [0.12 0.47 0.71];  % decreasing = blue

deg = char(176);

lat_list = [-30.5 -10.5 0.5 10.5:10:60.5];

% ---- constants for shading ----
xgap = [2017+7/12 2018+7/12];

sub_num = 0;

for lat_temp = lat_list
    sub_num = sub_num + 1;
    ax = nexttile(tlo, sub_num);
    hold(ax,'on');

    % ---- indices ----
    Lat_ind = find(abs(lat_ACCESS-lat_temp) == min(abs(lat_ACCESS-lat_temp)), 1);
    ri      = DBNN_rho_ind_mid(Lat_ind);

    % ---- axes limits early (so shading uses correct y-range) ----
    xlim(ax, t_range);
    ylim(ax, [12 22]);

    % ---- GRACE gap shading (behind everything) ----
    yl = ylim(ax);
    pgap = patch(ax, ...
        [xgap(1) xgap(2) xgap(2) xgap(1)], ...
        [yl(1)   yl(1)   yl(2)   yl(2)], ...
        'cyan', 'EdgeColor','none', 'FaceAlpha', alphaGap);
    uistack(pgap,'bottom');

    % ---- uncertainty band (behind lines) ----
    y = pred_strength_mid(:,Lat_ind);
    u = uncertainty_mid(:,Lat_ind);

    tvec  = t_pred(:);
    upper = y(:) + u(:);
    lower = y(:) - u(:);

    pband = patch(ax, [tvec; flipud(tvec)], [upper; flipud(lower)], ...
        Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);
    uistack(pband,'bottom');

    % ---- mean pred line ----
    plot(ax, t_pred, y, '-', 'LineWidth', LWmean, 'Color', Pred_color);

    % ---- trend line + annotation (keep your significance test) ----
    isSig = corr_pval(ri,Lat_ind) > (1 - significant);

    if isSig
        slope = trend_mean(ri,Lat_ind);
        b0    = intercepts_mean(ri,Lat_ind);
        trend_line = tvec*slope + b0;

        trendColor = trend_up_color;
        if slope < 0
            trendColor = trend_down_color;
        end

        plot(ax, tvec, trend_line, '--', 'LineWidth', LWtrend, 'Color', trendColor);

        t1 = trend_ci95(1,ri,Lat_ind);
        t2 = trend_ci95(2,ri,Lat_ind);
        text(ax, 0.45, 0.92, sprintf('Trend = [%.2f, %.2f] Sv yr^{-1}', t1, t2), ...
            'Units','normalized', 'FontName',fontname, 'FontSize',fontsize, ...
            'Interpreter','tex', 'Color', trendColor);
    else
        text(ax, 0.60, 0.92, 'Trend not significant', ...
            'Units','normalized', 'FontName',fontname, 'FontSize',fontsize, ...
            'Interpreter','none', 'Color', [0.35 0.35 0.35]);
    end

    % ---- panel annotation (top-left, bold) ----
    if lat_temp > 0
        ttl = sprintf('AMOC at %.1f%cN', lat_temp, deg);
    else
        ttl = sprintf('AMOC at %.1f%cS', abs(lat_temp), deg);
    end
    text(ax, 0.02, 0.98, ttl, 'Units','normalized', ...
        'FontName',fontname, 'FontSize',fontsize, 'FontWeight','bold', ...
        'Interpreter','none', 'VerticalAlignment','top', 'Color',[0.1 0.1 0.1]);

    % ---- ticks ----
    xticks(ax, 2004:4:2024);

    % show x tick labels only on bottom row
    if sub_num < 7
        ax.XTickLabel = [];
    else
        xlabel(ax,'Year','FontName',fontname,'FontSize',fontsize,'Interpreter','none');
    end

    % show y label only on left column
    if ismember(sub_num, [1 4 7])
        ylabel(ax,'\Psi (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');
    else
        ax.YTickLabel = [];
    end

    % ---- clean axes (Nature-ish) ----
    ax.FontName   = fontname;
    ax.FontSize   = fontsize;
    ax.LineWidth  = LW;
    ax.TickDir    = 'out';
    ax.Box        = 'off';
    ax.Layer      = 'top';
    ax.TickLength = [0.012 0.012];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';

    ax.YGrid = 'on';
    ax.XGrid = 'off';
    ax.GridAlpha = 0.12;
end

exportgraphics(figH, fullfile(model_dir_baseline,'RealWorld','AMOC_recons_multi_Lat_v3.png'), 'Resolution',300);


%% line plot - SOMOC (multi-lat)  [mid-depth + abyssal, same style as AMOC-final]
fontsize = 15;
LW = 1;

figH = figure(2);
set(figH,'Position',[100 100 1600 600],'Color','w','Renderer','painters');
tlo = tiledlayout(figH, 3, 3, 'TileSpacing','compact', 'Padding','compact');

% ---- style ----
fontname  = 'Arial';
LWmean    = LW + 1.2;
LWtrend   = LW + 0.6;
LWother   = LW + 0.6;
alphaBand = 0.18;
alphaGap  = 0.08;

Pred_color       = [0 0 0];           % DBNN mean
ECCO_color       = [0.35 0.35 0.35];  % ECCO (gray)
trend_up_color   = [0.85 0.2 0.2];    % increasing = red
trend_down_color = [0.12 0.47 0.71];  % decreasing = blue

deg  = char(176);
xgap = [2017+7/12 2018+7/12];

% ---- latitude lists ----
lat_list_mid  = [-55.5 -45.5 -35.5];
lat_list_abys = [-70.5 -60.5 -50.5];

% ---- per-panel y-limits (match your originals) ----
mid_ylim  = {[0 20], [10 30], [5 20]};
mid_ytick = {0:5:30, 10:5:30, 0:5:30};

abys_ylim  = {[0 15], [-5 15], [5 20]};
abys_ytick = {-5:5:30, -5:5:30, -5:5:30};

sub_num = 0;

%%% ===================== MID-DEPTH (row 1) =====================
for j = 1:numel(lat_list_mid)
    lat_temp = lat_list_mid(j);
    sub_num = sub_num + 1;

    ax = nexttile(tlo, sub_num);
    hold(ax,'on');

    % indices
    Lat_ind = find(abs(lat_ACCESS - lat_temp) == min(abs(lat_ACCESS - lat_temp)), 1);
    ri      = DBNN_rho_ind_mid(Lat_ind);

    % axes limits early (for shading)
    xlim(ax, t_range);
    ylim(ax, mid_ylim{j});
    yticks(ax, mid_ytick{j});

    % GRACE gap shading
    yl = ylim(ax);
    pgap = patch(ax, [xgap(1) xgap(2) xgap(2) xgap(1)], [yl(1) yl(1) yl(2) yl(2)], ...
        'cyan', 'EdgeColor','none', 'FaceAlpha', alphaGap);
    uistack(pgap,'bottom');

    % uncertainty band + mean
    y = pred_strength_mid(:,Lat_ind);
    u = uncertainty_mid(:,Lat_ind);
    tvec  = t_pred(:);
    upper = y(:) + u(:);
    lower = y(:) - u(:);

    pband = patch(ax, [tvec; flipud(tvec)], [upper; flipud(lower)], ...
        Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);
    uistack(pband,'bottom');

    plot(ax, t_pred, y, '-', 'LineWidth', LWmean, 'Color', Pred_color);

    % ECCO
    % Lat_ind_ecco = find(abs(lat_ecco - lat_temp) == min(abs(lat_ecco - lat_temp)), 1);
    % plot(ax, t_ecco, ECCO_strength_SOMOC_mid(:,Lat_ind_ecco), '-', ...
    %     'LineWidth', LWother, 'Color', ECCO_color);

    % trend + annotation (keep your significance test)
    isSig = corr_pval(ri,Lat_ind) > (1 - significant);
    if isSig
        slope = trend_mean(ri,Lat_ind);
        b0    = intercepts_mean(ri,Lat_ind);
        trend_line = tvec*slope + b0;

        % choose trend color by sign (in plotted coordinates)
        trendColor = trend_up_color;
        if slope < 0, trendColor = trend_down_color; end

        plot(ax, tvec, trend_line, '--', 'LineWidth', LWtrend, 'Color', trendColor);

        t1 = trend_ci95(1,ri,Lat_ind);
        t2 = trend_ci95(2,ri,Lat_ind);
        text(ax, 0.45, 0.12, sprintf('Trend = [%.2f, %.2f] Sv yr^{-1}', t1, t2), ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize, ...
            'Interpreter','tex', 'Color', trendColor);
    else
        text(ax, 0.60, 0.1, 'Trend not significant', ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize, ...
            'Interpreter','none', 'Color',[0.35 0.35 0.35]);
    end

    % top-left annotation (bold)
    ttl = sprintf('Mid-depth SMOC at %.1f%cS', abs(lat_temp), deg);
    text(ax, 0.02, 0.98, ttl, 'Units','normalized', ...
        'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
        'Interpreter','none', 'VerticalAlignment','top', 'Color',[0.1 0.1 0.1]);

    % ticks/labels
    xticks(ax, 2004:4:2024);
    ax.XTickLabel = [];  % hide for row 1




    if ismember(sub_num, [1 4 7])  % left column
        ylabel(ax,'\Psi (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');
    else
        % ax.YTickLabel = [];
    end

    % styling
    ax.FontName   = fontname;
    ax.FontSize   = fontsize;
    ax.LineWidth  = LW;
    ax.TickDir    = 'out';
    ax.Box        = 'off';
    ax.Layer      = 'top';
    ax.TickLength = [0.012 0.012];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.YGrid      = 'on';
    ax.XGrid      = 'off';
    ax.GridAlpha  = 0.12;
end

%%% ===================== ABYSSAL (row 2) =====================
for j = 1:numel(lat_list_abys)
    lat_temp = lat_list_abys(j);
    sub_num = sub_num + 1;

    ax = nexttile(tlo, sub_num);
    hold(ax,'on');

    % indices
    Lat_ind = find(abs(lat_ACCESS - lat_temp) == min(abs(lat_ACCESS - lat_temp)), 1);
    ri      = DBNN_rho_ind_abys(Lat_ind);

    % axes limits early (for shading)
    xlim(ax, t_range);
    ylim(ax, abys_ylim{j});
    yticks(ax, abys_ytick{j});

    % GRACE gap shading
    yl = ylim(ax);
    pgap = patch(ax, [xgap(1) xgap(2) xgap(2) xgap(1)], [yl(1) yl(1) yl(2) yl(2)], ...
        'cyan', 'EdgeColor','none', 'FaceAlpha', alphaGap);
    uistack(pgap,'bottom');

    % sign convention: plot -pred_strength_abys (as in your script)
    y_raw = pred_strength_abys(:,Lat_ind);
    u_raw = uncertainty_abys(:,Lat_ind);

    y = -y_raw;
    u =  u_raw;

    tvec  = t_pred(:);
    upper = y(:) + u(:);
    lower = y(:) - u(:);

    pband = patch(ax, [tvec; flipud(tvec)], [upper; flipud(lower)], ...
        Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);
    uistack(pband,'bottom');

    plot(ax, t_pred, y, '-', 'LineWidth', LWmean, 'Color', Pred_color);

    % ECCO (also negated to match sign convention)
    % Lat_ind_ecco = find(abs(lat_ecco - lat_temp) == min(abs(lat_ecco - lat_temp)), 1);
    % plot(ax, t_ecco, -ECCO_strength_SOMOC_abys(:,Lat_ind_ecco), '-', ...
    %     'LineWidth', LWother, 'Color', ECCO_color);

    % trend + annotation (keep your significance test)
    isSig = corr_pval(ri,Lat_ind) > (1 - significant);
    if isSig
        slope_raw = trend_mean(ri,Lat_ind);
        b0_raw    = intercepts_mean(ri,Lat_ind);

        % trend in plotted coordinates (negated)
        trend_line = -(tvec*slope_raw + b0_raw);

        % choose color by sign in plotted coordinates
        slope_plot = -slope_raw;
        trendColor = trend_up_color;
        if slope_plot < 0, trendColor = trend_down_color; end

        plot(ax, tvec, trend_line, '--', 'LineWidth', LWtrend, 'Color', trendColor);

        % CI in plotted coordinates (negated; keep low<high)
        t1 = trend_ci95(1,ri,Lat_ind);
        t2 = trend_ci95(2,ri,Lat_ind);
        t1p = -t2;  % flip order
        t2p = -t1;

        text(ax, 0.45, 0.12, sprintf('Trend = [%.2f, %.2f] Sv yr^{-1}', t1p, t2p), ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize, ...
            'Interpreter','tex', 'Color', trendColor);
    else
        text(ax, 0.60, 0.1, 'Trend not significant', ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize, ...
            'Interpreter','none', 'Color',[0.35 0.35 0.35]);
    end

    % top-left annotation (bold)
    ttl = sprintf('Abyssal SMOC at %.1f%cS', abs(lat_temp), deg);
    text(ax, 0.02, 0.98, ttl, 'Units','normalized', ...
        'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
        'Interpreter','none', 'VerticalAlignment','top', 'Color',[0.1 0.1 0.1]);

    % ticks/labels
    xticks(ax, 2004:4:2024);
    xlabel(ax,'Year','FontName',fontname,'FontSize',fontsize,'Interpreter','none'); % row 2 is bottom-used

    if ismember(sub_num, [1 4 7])  % left column (tile 4 is first of abyssal row)
        ylabel(ax,'-\Psi (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');
    else
        % ax.YTickLabel = [];
    end

    % styling
    ax.FontName   = fontname;
    ax.FontSize   = fontsize;
    ax.LineWidth  = LW;
    ax.TickDir    = 'out';
    ax.Box        = 'off';
    ax.Layer      = 'top';
    ax.TickLength = [0.012 0.012];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.YGrid      = 'on';
    ax.XGrid      = 'off';
    ax.GridAlpha  = 0.12;
end

% (Row 3 remains empty by design; keeps 3x3 layout consistent with AMOC)

exportgraphics(figH, fullfile(model_dir_baseline,'RealWorld','SOMOC_recons_multi_Lat_v3.png'), 'Resolution',300);