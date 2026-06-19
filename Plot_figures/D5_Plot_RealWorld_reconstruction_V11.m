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
basedir = 'E:\Analysis2026\ACCESS_hist+SSP126\ASMOC\results_LPF2Year';
rapidRoot = 'E:\Data_RAPID';
eccoRootV4r3 = 'E:\Data_ECCO\ECCOV4r3';
eccoRootV4r4 = 'E:\Data_ECCO\ECCOV4r4';

addRequiredPath(toolboxRoot, 'MATLAB toolbox root');
addRequiredPath(codeRoot, 'CMIP6 code root');

CT= cbrewer('qual','Set1',8);
SmoothParam = 5;
SmoothMethod = 'moving';

Pred_color = [0 0 0];

fontsize = 18;
LW = 1.5;

%%%% remove the first and last parts of the time series to avoid filter edge effect
edge = 12;   % change this number as needed

ECCO_v4r4 = 0;

%%
    % NNname = 'FullDepth_PCAinY50_ResNet_Neur128x64x128_5foldCV_Reg0.01Drop0.2_LPF2Year'
    % model_dir_baseline = fullfile(basedir, NNname, covariate_name);
    % cd(model_dir_baseline)
    % cd('./RealWorld')
%%

% =========================================================
% 1) Baseline
% =========================================================
baseline_tests = {
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
    };

% =========================================================
% 2) PCA dimension tests
% =========================================================
pca_tests = {
    'FullDepth_PCAinY16_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year';
    % 'FullDepth_PCAinY32_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_PCAinY64_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_PCAinY128_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year',
    'FullDepth_PCAinY16_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY32_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY64_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY128_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
    };

% =========================================================
% 3) Activation-function tests
% =========================================================
activation_tests = {
    % 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_linearActivation_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_reluActivation_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_geluActivation_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_sigmoidActivation_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_linearActivation_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_reluActivation_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_geluActivation_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_sigmoidActivation_LPF2Year';
    };

% =========================================================
% 4) Regularization tests
% =========================================================
reg_tests = {
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.0001_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.001_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.1_LPF2Year';
    };

% =========================================================
% 5) Dropout tests
% =========================================================
dropout_tests = {
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.0001Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.001Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.1Drop0.2_LPF2Year';
    };

% =========================================================
% 6) Architecture tests
% =========================================================
arch_tests = {
    % 'FullDepth_PCAinY50_ResNet_Neur32_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur64_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur128x64_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur128x64x128_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur256_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur512_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur512x256_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur512x256x128_5foldCV_Reg0.01_LPF2Year',
    % 'FullDepth_PCAinY50_ResNet_Neur512x64x512_5foldCV_Reg0.01_LPF2Year',
    'FullDepth_PCAinY50_ResNet_Neur32_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur64_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur128x64_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur128x64x128_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur256_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128_5foldCV_Reg0.01Drop0.2_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x64x512_5foldCV_Reg0.01Drop0.2_LPF2Year';
    };

% =========================================================
% 7) MAE-loss tests
% =========================================================
maeloss_tests = {
    'FullDepth_PCAinY50_ResNet_Neur64_5foldCV_Reg0.01_maeloss_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur256_5foldCV_Reg0.01_maeloss_LPF2Year';
    'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01_maeloss_LPF2Year';
    };

% =========================================================
% Combine selected experiment groups
% =========================================================
NNnameALL = baseline_tests;
% NNnameALL = [NNnameALL; pca_tests];
% NNnameALL = [NNnameALL; activation_tests];
% NNnameALL = [NNnameALL; reg_tests];
% NNnameALL = [NNnameALL; dropout_tests];
% NNnameALL = [NNnameALL; arch_tests];
% NNnameALL = [NNnameALL; maeloss_tests];

%% Load DBNN prediction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% covariate_name = 'ssh_mascon_V5+uas_mascon_V5';%+ssh_mascon_V5+uas_mascon_V5
covariate_name = 'obp_mascon_V5+ssh_mascon_V5+uas_mascon_V5';%+ssh_mascon_V5+uas_mascon_V5
for ii = 1:numel(NNnameALL)

    close all

    NNname = NNnameALL{ii};
    % NNname = 'FullDepth_PCAinY50_ResNet_Neur512x256x128x64_5foldCV_Reg0.01Drop0.2_LPF2Year'; %
    model_dir_baseline = fullfile(basedir, NNname, covariate_name);
    realWorldDir = fullfile(model_dir_baseline, 'RealWorld');
    ensureExistingFolder(model_dir_baseline, 'Baseline model directory');
    ensureExistingFolder(realWorldDir, 'Real-world output directory');
    load(resolveExistingMatFile(fullfile(realWorldDir, 'Pred_RealWorld'), 'Real-world prediction file'));
    %%%% remove the first and last parts of the time series to avoid filter edge effect
    pred_yz = pred_yz(edge+1:end-edge,:,:);
    pred_yz_std = pred_yz_std(edge+1:end-edge,:,:);
    rho2 = rho2 - 1000;
    [Nsamps, Nrho, Nlat0] = size(pred_yz);
    lat_ACCESS = lat;
    clear lat

    t_pred = 2002 + 4/12 + 1 + (0:Nsamps-1)/12;
    t_range = [t_pred(1) t_pred(end)];


    %%% uncertainty based on SSP245 test

    load(resolveExistingMatFile(fullfile(model_dir_baseline, 'rmse_SSP245'), ...
        'SSP245 RMSE file'))

    NN_rmse = rmse_yz;
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
    load(resolveExistingMatFile(fullfile(rapidRoot, 'Rapid_LPF'), 'RAPID LPF file'))
    %%%% remove the first and last 24 months to avoid filter edge effect
    RAPID_monthly_LPF_web = RAPID_monthly_LPF(edge+1:end-edge);
    % t_RAPID = t_year(edge+1:end-edge);
    %%%
    %
    %
    %
    load(resolveExistingMatFile(fullfile(rapidRoot, 'Rapid_FullDepth_LPF'), ...
        'RAPID full-depth LPF file'))
    %%% remove the first and last 24 months to avoid filter edge effect
    RAPID_depth_LPF= stream_depth_LPF(edge+1:end-edge,:);
    RAPID_depth_LPF(:,1:find(depth>500,1)-1)=nan;
    ind_max = find(mean(RAPID_depth_LPF) == max(mean(RAPID_depth_LPF)));
    depth(ind_max);
    % RAPID_monthly_LPF = max(RAPID_depth_LPF,[],2)';
    RAPID_monthly_LPF_z =RAPID_depth_LPF(:,ind_max)';
    % t_RAPID = t_year(edge+1:end-edge);
    %%%


    load(resolveExistingMatFile(fullfile(rapidRoot, 'Rapid_FullDepth_LPF'), ...
        'RAPID full-depth LPF file'))
    %%%% remove the first and last 24 months to avoid filter edge effect
    RAPID_sigma2_LPF= stream_sigma2_LPF(edge+1:end-edge,:);
    RAPID_sigma2_LPF(:,1:find(sigma2>1035,1)-1)=nan;
    ind_max = find(mean(RAPID_sigma2_LPF) == max(mean(RAPID_sigma2_LPF)));
    sigma2(ind_max);
    RAPID_monthly_LPF_sigma2 =RAPID_sigma2_LPF(:,ind_max)';


    RAPID_sigma0_LPF= stream_sigma0_LPF(edge+1:end-edge,:);
    % RAPID_sigma0_LPF(:,1:find(sigma0>1035,1)-1)=nan;
    ind_max = find(mean(RAPID_sigma0_LPF) == max(mean(RAPID_sigma0_LPF)));
    sigma0(ind_max);
    RAPID_monthly_LPF_sigma0 =RAPID_sigma0_LPF(:,ind_max)';

    t_RAPID = t_year(edge+1:end-edge);
    %%%

    corr(RAPID_monthly_LPF_sigma2',RAPID_monthly_LPF_z');


    %% ECCO
    %%% V4r3
    if useEccoV4r4(ECCO_v4r4)
        eccoPsiFile = resolveExistingMatFile(fullfile(eccoRootV4r4, 'PSI_AMOC_LPF'), ...
            'ECCO v4r4 MOC file');
        eccoLatFile = fullfile(eccoRootV4r4, 'myproducts_monthly', 'PSI.mat');
        if ~isfile(eccoLatFile)
            % Fall back to the v4r3 grid file when the v4r4 PSI grid file is absent.
            eccoLatFile = resolveExistingMatFile(fullfile(eccoRootV4r3, 'myproducts_monthly', 'PSI.mat'), ...
                'ECCO latitude grid file');
        end
    else
        eccoPsiFile = resolveExistingMatFile(fullfile(eccoRootV4r3, 'PSI_LPF'), ...
            'ECCO v4r3 MOC file');
        eccoLatFile = resolveExistingMatFile(fullfile(eccoRootV4r3, 'myproducts_monthly', 'PSI.mat'), ...
            'ECCO latitude grid file');
    end
    eccoPsiData = load(eccoPsiFile);
    eccoLatData = load(eccoLatFile);
    MOC_ecco_LPF = eccoPsiData.MOC_ecco_LPF;
    t_ecco = eccoPsiData.t_year_ecco;
    lat = eccoLatData.lat;


    MOC_ecco_LPF = MOC_ecco_LPF-MOC_ecco_LPF(:,end,:); % [time,rho,lat]


    % MOC_ecco_LPF(:,1:42,:) = 0; %ignore sigma2<35
    lat_ecco = 0.5*( lat(1:end-1)+ lat(2:end));
    PSI_ECCO = 0.5* (MOC_ecco_LPF(edge+1:end-edge,:,1:end-1)+MOC_ecco_LPF(edge+1:end-edge,:,2:end));
    t_ecco = t_ecco(edge+1:end-edge);

    % MOC strength
    PSI_ECCO_mean = squeeze(mean(PSI_ECCO,1));
    [~,rho_ind_mean] = max(PSI_ECCO_mean,[],1);
    % rho_ind_mean = smooth(lat_ecco,rho_ind_mean,SmoothParam,SmoothMethod);
    % rho_ind_mean = round(rho_ind_mean);
    ECCO_strength_mid = zeros(length(PSI_ECCO(:,1,1)),length(PSI_ECCO(1,1,:)));
    for jj = 1:length(PSI_ECCO(1,1,:))
        ECCO_strength_mid(:,jj) = PSI_ECCO(:,rho_ind_mean(jj),jj);
    end
    clear rho_ind_mean
    ECCO_strength_mid_test = max( PSI_ECCO(:,:,116),[],2);


    [~,rho_ind_mean_abys] = min(PSI_ECCO_mean,[],1);
    % rho_ind_mean = smooth(lat_ecco,rho_ind_mean,SmoothParam,SmoothMethod);
    % rho_ind_mean = round(rho_ind_mean);
    ECCO_strength_abys = zeros(length(PSI_ECCO(:,1,1)),length(PSI_ECCO(1,1,:)));
    for jj = 1:length(PSI_ECCO(1,1,:))
        ECCO_strength_abys(:,jj) = PSI_ECCO(:,rho_ind_mean_abys(jj),jj);
    end
    clear rho_ind_mean_abys



    %% polished

    lat_temp = 26.5;

    % ---------- NCC-ish figure geometry (2-column width ~18.3 cm) ----------
    figW = 18.3;   % cm
    figH = 6;    % cm
    figure('Units','centimeters','Position',[2 2 figW figH], ...
        'Color','w','Renderer','painters');

    ax = axes('Position',[0.065 0.22 0.92 0.72]);  % compact margins
    hold(ax,'on');

    % ---------- Colors (muted, colorblind-friendly) ----------
    cRAPID = [0.85 0.33 0.10];   % orange-red
    cDBNN  = [0.10 0.10 0.10];   % near-black
    cECCO  = [0.00 0.45 0.74];   % blue

    lw_main = 1.6;
    lw_aux  = 1.2;
    alpha_obs  = 0.18;
    alpha_pred = 0.14;

    % ===================== RAPID =====================
    h1 = plot(ax,t_RAPID, RAPID_monthly_LPF_web, '-', ...
        'Color', cRAPID, 'LineWidth', lw_main);

    % IMPORTANT BUG FIX: use logical indexing with & (not 2005<t<=2006)
    RAPID_uncertainty = 0.9*ones(size(t_RAPID));
    RAPID_uncertainty(t_RAPID>2005 & t_RAPID<=2006) = 1.0;
    RAPID_uncertainty(t_RAPID>2007 & t_RAPID<=2008) = 1.3;
    RAPID_uncertainty = smoothn(RAPID_uncertainty,'robust');
    RAPID_upper = RAPID_monthly_LPF_web(:) + RAPID_uncertainty(:);
    RAPID_lower = RAPID_monthly_LPF_web(:) - RAPID_uncertainty(:);

    patch(ax, [t_RAPID(:); flipud(t_RAPID(:))], ...
        [RAPID_upper; flipud(RAPID_lower)], ...
        cRAPID, 'FaceAlpha', alpha_obs, 'EdgeColor','none');

    % ===================== DBNN =====================
    Lat_ind = find(abs(lat_ACCESS-lat_temp)==min(abs(lat_ACCESS-lat_temp)),1);

    y_pred = pred_strength_mid(:,Lat_ind);
    u_pred = uncertainty_mid(:,Lat_ind);
    pred_upper = y_pred(:) + u_pred(:);
    pred_lower = y_pred(:) - u_pred(:);

    patch(ax, [t_pred(:); flipud(t_pred(:))], ...
        [pred_upper; flipud(pred_lower)], ...
        cDBNN, 'FaceAlpha', alpha_pred, 'EdgeColor','none');

    h2 = plot(ax, t_pred, y_pred, '-', ...
        'Color', cDBNN, 'LineWidth', lw_main);

    % ===================== ECCO =====================
    Lat_ind_ecco = find(abs(lat_ecco-lat_temp)==min(abs(lat_ecco-lat_temp)),1);
    h3 = plot(ax, t_ecco, ECCO_strength_mid(:,Lat_ind_ecco), '--', ...
        'Color', cECCO, 'LineWidth', lw_aux);

    % ===================== Gap shading (subtle) =====================
    yl = [12 20.5];
    xgap = [2017+7/12 2018+7/12];
    patch(ax, [xgap(1) xgap(2) xgap(2) xgap(1)], ...
        [yl(1)   yl(1)   yl(2)   yl(2)], ...
        'cyan', 'FaceAlpha', 0.08, 'EdgeColor','none');  % light gray band

    % ===================== Axes styling (NCC-like) =====================
    ax.FontName   = 'Arial';
    ax.FontSize   = 8;
    ax.LineWidth  = 0.75;
    ax.TickDir    = 'out';
    ax.TickLength = [0.012 0.012];
    ax.Box        = 'off';
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';

    ax.YGrid      = 'on';
    ax.XGrid      = 'off';
    ax.GridAlpha  = 0.12;

    xlim(ax, t_range);
    ylim(ax, yl);

    xlabel(ax,'Year','FontName','Arial','FontSize',9);
    ylabel(ax,'AMOC strength (Sv)','FontName','Arial','FontSize',9);

    % ===================== Statistics (clean + correct overlap) =====================
    t0 = max(t_RAPID(1), t_pred(1));
    t1 = min(t_RAPID(end), t_pred(end));
    iR = (t_RAPID>=t0 & t_RAPID<=t1);
    iP = (t_pred >=t0 & t_pred <=t1);

    % Make sure lengths match; if not, interpolate pred onto RAPID times
    if sum(iR) ~= sum(iP) || any(abs(t_RAPID(iR) - t_pred(iP)) > 1e-8)
        y_pred_on_R = interp1(t_pred(iP), y_pred(iP), t_RAPID(iR), 'linear', 'extrap');
    else
        y_pred_on_R = y_pred(iP);
    end

    r = corr(RAPID_monthly_LPF_web(iR)', y_pred_on_R(:), 'rows','complete');
    rmse = sqrt(mean((RAPID_monthly_LPF_web(iR)' - y_pred_on_R(:)).^2, 'omitnan'));

    text(ax, 0.99, 1.03, sprintf('r = %.2f, RMSE = %.2f Sv', r, rmse), ...
        'Units','normalized', 'HorizontalAlignment','right', 'VerticalAlignment','top', ...
        'FontName','Arial', 'FontSize',9, 'Color',[0 0 0]);

    % ===================== Legend (small, no border) =====================
    lgd = legend(ax, [h1 h2 h3], {'RAPID','NeurMOC','ECCO'}, ...
        'Location','north', 'Orientation','horizontal', 'Box','off','Position',[0.319239491405735 0.891987885623379 0.357917566426145 0.0726872230285065]);
    lgd.FontName = 'Arial';
    lgd.FontSize = 8;

    % ===================== Export (submission-friendly) =====================
    exportgraphics(gcf, fullfile(model_dir_baseline,'RealWorld','Figure4_V4.pdf'), ...
        'ContentType','vector');   % best for journals
    exportgraphics(gcf, fullfile(model_dir_baseline,'RealWorld','AA_RAPID_cmp.png'), ...
        'Resolution', 300);


    %% compute trend

    significant = 0.05;

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
        fprintf('Processing density index %d of %d.\n', k, nk);
        for j = 1:nj
            y = squeeze(pred_yz(:,k,j)); % time series
            sigma = NN_rmse(:,k,j);

            slopes = nan(nMC,1);
            intercepts = nan(nMC,1);
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



%% save NeroMOC data

 NeroMOC = pred_yz;
 NeroMOC_uncertainty = NN_rmse;
 NeroMOC_time = t_pred;
 GRACE_Gap_TimeRang = [2017+7/12 2018+7/12];
 NeroMOC_latitude = lat_ACCESS;
 NeroMOC_density=rho2;
 NeroMOC_trend_mean=trend_mean;
 NeroMOC_trend_ci95=trend_ci95;
 
 NeroMOC_trend_significant = double(corr_pval > (1-significant));

save(fullfile(realWorldDir, 'NeroMOC_data.mat'), "NeroMOC","NeroMOC_uncertainty","NeroMOC_time", ...
    "GRACE_Gap_TimeRang","NeroMOC_latitude","NeroMOC_density", ...
    "NeroMOC_trend_mean","NeroMOC_trend_ci95","NeroMOC_trend_significant")
    %% plot the distribution of the trend
    %%% plot the distribution of the trend (polished, same layout/logic)


    % sign_mask = squeeze(sign(mean(pred_yz(1:120,:,:),1)));
    sign_mask =  1;
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

    bg_gray = [0.90 0.90 0.90];   % lighter than 0.8 for print

    % ---- colormap (diverging, centered at 0) ----
    clim_trend = [-0.4 0.4];
    cmp = cbrewer('div','RdBu',40);
    cmp = flipud(cmp);            % blue = negative, red = positive (typical)
    % Make NaNs look like background (if any)
    trend_plot = trend_mean;      % modify here if you want masking

    % ---- masks (compute once) ----
    MaskNeg = squeeze(mean(pred_yz(1:120,:,:),1)) < 0;        % [rho x lat]
    % MaskNeg = squeeze(mean(pred_yz(end-119:end,:,:),1)) < 0;        % [rho x lat]
    % MaskNeg = squeeze(mean(pred_yz,1)) < 0;        % [rho x lat]
    maskInsig = double(corr_pval < (1-significant));          % keep your test as-is

    % downsample indices for dots
    [JJneg, IIneg] = find(MaskNeg');  % JJneg->lat index, IIneg->rho index
    stepDots = 1;                     % 1=all, 2=sparser
    JJneg = JJneg(1:stepDots:end);
    IIneg = IIneg(1:stepDots:end);

    % for contourf y grid (your half-step shift)
    rho_edge = rho2 - mean(diff(rho2))/2;

    % ---- figure ----
    fh = figure('Position',framepos,'Color','w','Renderer','painters');

    %%% ===================== LEFT: SOMOC =====================
    ax1 = axes('Position',axpos(1,:)); hold(ax1,'on');

    imagesc(ax1, lat_ACCESS, rho2, trend_plot.*sign_mask);
    set(ax1,'YDir','reverse');
    colormap(ax1, cmp);
    clim(ax1, clim_trend);

    axis(ax1, [-74 -34 35 (rho2(end)+rho2(end-1))/2]);
    yticks(ax1, 35:0.5:38);

    % ticks/labels
    xt = -75:5:-35;
    xticks(ax1, xt);
    lbl = arrayfun(@(x) '', xt, 'UniformOutput', false);
    lbl(xt==-70) = {sprintf('70%cS',deg)};
    lbl(xt==-60) = {sprintf('60%cS',deg)};
    lbl(xt==-50) = {sprintf('50%cS',deg)};
    lbl(xt==-40) = {sprintf('40%cS',deg)};
    xticklabels(ax1, lbl);

    yt = -35:0.5:37.5;
    yticks(ax1, yt);
    ylabel(ax1, 'Density \sigma_2 (kg/m^3)', ...
        'FontSize',fontsize,'Interpreter','tex','FontName',fontname);

    % axis styling
    set(ax1,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
        'TickDir','out','Box','off','Layer','top', ...
        'TickLength',[0.008667 0], ...
        'XTickLabelRotation', 0,...
        'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
    ax1.Color = bg_gray;

    % negative MOC dots
    scatter(ax1, lat_ACCESS(JJneg), rho2(IIneg), 8, 'w', 'filled', ...
        'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

    % colorbar (single, shared)
    cb = colorbar(ax1,'Position',cbpos,'LineWidth',LW,'FontSize',fontsize);
    cb.TickDirection = 'out';
    cb.Box = 'off';
    cb.FontName = fontname;
    title(cb,'(Sv yr^{-1})','FontSize',fontsize,'FontName',fontname,'Interpreter','tex');


    % insignificant trend hatch (keep your hatchfill2)
    [~, h1hat] = contourf(ax1, lat_ACCESS, rho_edge, maskInsig, [1 1], 'EdgeColor','none');
    hatchfill2(h1hat,'single','HatchAngle',50,'HatchColor',[0 0 0], ...
        'FaceColor','none','HatchDensity',25*1.2,'HatchLineWidth',max(0.2,LW-0.8));

    % rho lines
    plot(ax1, lat_ACCESS, rho_DBNN,      '--k', 'LineWidth', LW+0.5);
    plot(ax1, lat_ACCESS, rho_DBNN_abys, '--k', 'LineWidth', LW+0.5);


    %%% ===================== RIGHT: AMOC =====================
    ax2 = axes('Position',axpos(2,:)); hold(ax2,'on');

    imagesc(ax2, lat_ACCESS, rho2, trend_plot.*sign_mask);
    set(ax2,'YDir','reverse');
    colormap(ax2, cmp);
    clim(ax2, clim_trend);

    axis(ax2, [-34 64 35 (rho2(end)+rho2(end-1))/2]);

    % ticks/labels
    xt = -35:5:75;
    xticks(ax2, xt);

    lbl = repmat({''}, size(xt));
    for i = 1:numel(xt)
        v = xt(i);
        if v == 0
            lbl{i} = sprintf('0%c', deg);
        elseif mod(abs(v),10)==0   % label every 10°
            if v < 0, lbl{i} = sprintf('%d%cS', abs(v), deg);
            else,     lbl{i} = sprintf('%d%cN', v, deg);
            end
        end
    end

    xticklabels(ax2, lbl);

    yt = -35:0.5:37.5;
    yticks(ax2, yt);
    yticklabels(ax2, {''});

    % axis styling
    set(ax2,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
        'TickDir','out','Box','off','Layer','top', ...
        'TickLength',[0.004 0], ...
        'XTickLabelRotation', 0,...
        'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
    ax2.Color = bg_gray;

    % negative MOC dots
    scatter(ax2, lat_ACCESS(JJneg), rho2(IIneg), 8, 'w', 'filled', ...
        'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

    % insignificant trend hatch (different angle/density like you had)
    [~, h2hat] = contourf(ax2, lat_ACCESS, rho_edge, maskInsig, [1 1], 'EdgeColor','none');
    hatchfill2(h2hat,'single','HatchAngle',70,'HatchColor',[0 0 0], ...
        'FaceColor','none','HatchDensity',50*1.2,'HatchLineWidth',max(0.2,LW-0.8));

    % rho line
    plot(ax2, lat_ACCESS, rho_DBNN, '--k', 'LineWidth', LW+0.5);


    text(ax1, 0.02, 0.98, 'SMOC', 'Units','normalized', ...
        'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
        'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
        'Color',[0.1 0.1 0.1]);

    text(ax2, 0.02, 0.98, 'AMOC', 'Units','normalized', ...
        'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
        'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
        'Color',[0.1 0.1 0.1]);

    %%% export
    exportgraphics(fh, fullfile(model_dir_baseline,'RealWorld','AA_trend_rec.png'), 'Resolution',300);

    %% epistemic uncertainty

    NN_epist_mean = squeeze(mean(NN_epist));

    % ---- layout ----
    framepos = [100 50 1300 600];
    axpos = zeros(5,4);
    axpos(1,:) = [0.08 0.2 0.25 0.6];
    axpos(2,:) = [0.34 0.2 0.6  0.6];
    cbpos      = [0.95 0.2 0.01 0.6];

    % ---- style ----
    fontsize = 18;
    LW = 1.5;
    fontname = 'Arial';
    deg = char(176);

    bg_gray = [0.90 0.90 0.90];   % lighter than 0.8 for print

    % ---- colormap (diverging, centered at 0) ----
    clim_plot= [0 2];
    % cmp = cbrewer('seq','Reds',40);
    cmp = cbrewer('div','Spectral',40);
    % cmp = cmocean('sha',40);
    cmp = flipud(cmp);            % blue = negative, red = positive (typical)
    % Make NaNs look like background (if any)
    trend_plot = trend_mean;      % modify here if you want masking

    % ---- masks (compute once) ----
    MaskNeg = squeeze(mean(pred_yz(1:120,:,:),1)) < 0;        % [rho x lat]

    % downsample indices for dots
    [JJneg, IIneg] = find(MaskNeg');  % JJneg->lat index, IIneg->rho index
    stepDots = 1;                     % 1=all, 2=sparser
    JJneg = JJneg(1:stepDots:end);
    IIneg = IIneg(1:stepDots:end);

    % for contourf y grid (your half-step shift)
    rho_edge = rho2 - mean(diff(rho2))/2;

    % ---- figure ----
    fh = figure('Position',framepos,'Color','w','Renderer','painters');

    %%% ===================== LEFT: SOMOC =====================
    ax1 = axes('Position',axpos(1,:)); hold(ax1,'on');

    imagesc(ax1, lat_ACCESS, rho2, NN_epist_mean);
    set(ax1,'YDir','reverse');
    colormap(ax1, cmp);
    clim(ax1, clim_plot);

    axis(ax1, [-74 -34 35 (rho2(end)+rho2(end-1))/2]);
    yticks(ax1, 35:0.5:38);

    % ticks/labels
    xt = -75:5:-35;
    xticks(ax1, xt);
    lbl = arrayfun(@(x) '', xt, 'UniformOutput', false);
    lbl(xt==-70) = {sprintf('70%cS',deg)};
    lbl(xt==-60) = {sprintf('60%cS',deg)};
    lbl(xt==-50) = {sprintf('50%cS',deg)};
    lbl(xt==-40) = {sprintf('40%cS',deg)};
    xticklabels(ax1, lbl);

    yt = -35:0.5:37.5;
    yticks(ax1, yt);
    ylabel(ax1, 'Density \sigma_2 (kg/m^3)', ...
        'FontSize',fontsize,'Interpreter','tex','FontName',fontname);

    % axis styling
    set(ax1,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
        'TickDir','out','Box','off','Layer','top', ...
        'TickLength',[0.008667 0], ...
        'XTickLabelRotation', 0,...
        'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
    ax1.Color = bg_gray;

    % negative MOC dots
    scatter(ax1, lat_ACCESS(JJneg), rho2(IIneg), 8, 'w', 'filled', ...
        'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);

    % colorbar (single, shared)
    cb = colorbar(ax1,'Position',cbpos,'LineWidth',LW,'FontSize',fontsize);
    cb.TickDirection = 'out';
    cb.Box = 'off';
    cb.FontName = fontname;
    title(cb,'(Sv)','FontSize',fontsize,'FontName',fontname,'Interpreter','tex');



    % rho lines
    plot(ax1, lat_ACCESS, rho_DBNN,      '--k', 'LineWidth', LW+0.5);
    plot(ax1, lat_ACCESS, rho_DBNN_abys, '--k', 'LineWidth', LW+0.5);


    %%% ===================== RIGHT: AMOC =====================
    ax2 = axes('Position',axpos(2,:)); hold(ax2,'on');

    imagesc(ax2, lat_ACCESS, rho2, NN_epist_mean);
    set(ax2,'YDir','reverse');
    colormap(ax2, cmp);
    clim(ax2, clim_plot);

    axis(ax2, [-34 64 35 (rho2(end)+rho2(end-1))/2]);

    % ticks/labels
    xt = -35:5:75;
    xticks(ax2, xt);

    lbl = repmat({''}, size(xt));
    for i = 1:numel(xt)
        v = xt(i);
        if v == 0
            lbl{i} = sprintf('0%c', deg);
        elseif mod(abs(v),10)==0   % label every 10°
            if v < 0, lbl{i} = sprintf('%d%cS', abs(v), deg);
            else,     lbl{i} = sprintf('%d%cN', v, deg);
            end
        end
    end

    xticklabels(ax2, lbl);

    yt = -35:0.5:37.5;
    yticks(ax2, yt);
    yticklabels(ax2, {''});

    % axis styling
    set(ax2,'FontName',fontname,'FontSize',fontsize,'LineWidth',LW, ...
        'TickDir','out','Box','off','Layer','top', ...
        'TickLength',[0.004 0], ...
        'XTickLabelRotation', 0,...
        'XMinorTick','on','YMinorTick','on','TickLabelInterpreter','none');
    ax2.Color = bg_gray;

    % negative MOC dots
    scatter(ax2, lat_ACCESS(JJneg), rho2(IIneg), 8, 'w', 'filled', ...
        'MarkerFaceAlpha',1,'MarkerEdgeAlpha',0);


    % rho line
    plot(ax2, lat_ACCESS, rho_DBNN, '--k', 'LineWidth', LW+0.5);


    title(ax1, 'SMOC', ...
        'FontSize',fontsize,'Interpreter','tex','FontName',fontname);
    title(ax2, 'AMOC', ...
        'FontSize',fontsize,'Interpreter','tex','FontName',fontname);
    % text(ax1, 0.02, 0.98, 'SMOC', 'Units','normalized', ...
    %     'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    %     'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    %     'Color',[0.1 0.1 0.1]);
    %
    % text(ax2, 0.02, 0.98, 'AMOC', 'Units','normalized', ...
    %     'FontName',fontname,'FontSize',fontsize,'FontWeight','bold', ...
    %     'Interpreter','none', 'VerticalAlignment','top','HorizontalAlignment','left', ...
    %     'Color',[0.1 0.1 0.1]);

    %%% export
    exportgraphics(fh, fullfile(model_dir_baseline,'RealWorld','RealWorld_EpistemicUncertainty_v3.png'), 'Resolution',300);





    %% line plot - AMOC

    fontsize = 16;

    framepos = [100 100  600   250];
    figure('Position',framepos);
    Pred_color =[0 0 0];


    %% line plot - AMOC (-20.5)
    Lat_plot = -20.5;

    figure('Position',framepos,'Color','w','Renderer','painters');
    ax = axes; hold(ax,'on');

    % ----- style -----
    fontname = 'Arial';
    LWmean   = LW + 1.2;
    LWtrend  = LW + 0.6;
    alphaBand = 0.18;
    alphaGap  = 0.08;

    % ----- index -----
    lat_temp = Lat_plot;
    Lat_ind = find(abs(lat_ACCESS-lat_temp) == min(abs(lat_ACCESS-lat_temp)),1);

    y = pred_strength_mid(:,Lat_ind);
    u = uncertainty_mid(:,Lat_ind);

    % ----- GRACE gap shading (put FIRST so it is behind everything) -----
    yl = [13 20];
    xgap = [2017+7/12 2018+7/12];
    pgap = patch(ax, [xgap(1) xgap(2) xgap(2) xgap(1)], [yl(1) yl(1) yl(2) yl(2)], ...
        'cyan', 'EdgeColor','none', 'FaceAlpha', alphaGap);
    uistack(pgap,'bottom');

    % ----- uncertainty band (patch first) -----
    upper = y(:) + u(:);
    lower = y(:) - u(:);
    pband = patch(ax, [t_pred(:); flipud(t_pred(:))], [upper; flipud(lower)], ...
        Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);


    uistack(pband,'bottom');  % keep under lines (still above gap is fine)

    % ----- mean line -----
    plot(ax, t_pred, y, '-', 'LineWidth', LWmean, 'Color', Pred_color);

    % ----- trend line -----
    slope = trend_mean(DBNN_rho_ind_mid(Lat_ind), Lat_ind);
    b0    = intercepts_mean(DBNN_rho_ind_mid(Lat_ind), Lat_ind);
    trend_line = t_pred(:)*slope + b0;

    plot(ax, t_pred, trend_line, '--', 'LineWidth', LWtrend, 'Color', [0.12 0.47 0.71]);

    % ----- axes -----
    xlim(ax, t_range);
    ylim(ax, yl);
    xticks(ax, 2004:4:2024);

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

    % ----- trend significance annotation -----
    isSig = corr_pval(DBNN_rho_ind_mid(Lat_ind),Lat_ind) > (1 - significant);

    if isSig
        t1 = trend_ci95(1, DBNN_rho_ind_mid(Lat_ind), Lat_ind);
        t2 = trend_ci95(2, DBNN_rho_ind_mid(Lat_ind), Lat_ind);
        text(ax, 0.45, 0.90, sprintf('Trend = [%.2f, %.2f] Sv yr^{-1}', t1, t2), ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize-1, ...
            'Interpreter','tex', 'Color',[0.12 0.47 0.71]);
    else
        text(ax, 0.60, 0.90, 'Trend not significant', ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize-1, ...
            'Interpreter','none', 'Color',[0.35 0.35 0.35]);
    end


    % export
    exportgraphics(gcf, fullfile(model_dir_baseline,'RealWorld','AA_AMOC20S.png'), 'Resolution',300);
    % exportgraphics(gcf, fullfile(model_dir_baseline,'RealWorld','AAAA4.pdf'), 'ContentType','vector');


    %% line plot - AMOC - lower limb
    Lat_plot = 26.5;

    figure('Position',framepos,'Color','w','Renderer','painters');
    ax = axes; hold(ax,'on');

    % ---- style ----
    fontname  = 'Arial';
    LWmean    = LW + 1.2;
    LWtrend   = LW + 0.6;
    alphaBand = 0.18;
    alphaGap  = 0.08;

    % ---- indices ----
    lat_temp = Lat_plot;
    Lat_ind  = find(abs(lat_ACCESS-lat_temp) == min(abs(lat_ACCESS-lat_temp)), 1);
    Rho_ind  = 16;

    y  = pred_yz(:,Rho_ind,Lat_ind);
    e  = NN_rmse(:,Rho_ind,Lat_ind);   % using RMSE as +/- envelope (as you do)

    % ---- axes limits (set early for shading) ----
    xlim(ax, t_range);
    ylim(ax, [-5 4]);

    % ---- GRACE gap shading (behind everything) ----
    yl = ylim(ax);
    xgap = [2017+7/12 2018+7/12];
    pgap = patch(ax, [xgap(1) xgap(2) xgap(2) xgap(1)], [yl(1) yl(1) yl(2) yl(2)], ...
        'cyan', 'EdgeColor','none', 'FaceAlpha', alphaGap);
    uistack(pgap,'bottom');



    % ---- uncertainty band (behind lines) ----
    upper = y(:) + e(:);
    lower = y(:) - e(:);
    pband = patch(ax, [t_pred(:); flipud(t_pred(:))], -[upper; flipud(lower)], ...
        Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);
    uistack(pband,'bottom');

    % ---- mean line ----
    plot(ax, [2000 2050], [0 0], '-', 'LineWidth', 0.5, 'Color', Pred_color);

    plot(ax, t_pred, -y, '-', 'LineWidth', LWmean, 'Color', Pred_color);

    % ---- trend line ----
    slope = trend_mean(Rho_ind,Lat_ind);
    b0    = intercepts_mean(Rho_ind,Lat_ind);
    trend_line = -t_pred(:)*slope - b0;
    plot(ax, t_pred, trend_line, '--', 'LineWidth', LWtrend, 'Color', [0.12 0.47 0.71]);

    % ---- labels ----
    xlabel(ax,'Year','FontName',fontname,'FontSize',fontsize,'Interpreter','none');
    ylabel(ax,'-\psi_{\sigma_2=36.94} (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');
    % kg/m^3
    % ---- ticks ----
    xticks(ax, 2004:4:2024);

    % ---- clean axes (Nature-ish) ----
    ax.FontName = fontname;
    ax.FontSize = fontsize;
    ax.LineWidth = LW;
    ax.TickDir = 'out';
    ax.Box = 'off';
    ax.Layer = 'top';
    ax.TickLength = [0.012 0.012];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';

    ax.YGrid = 'on';
    ax.XGrid = 'off';
    ax.GridAlpha = 0.12;

    % ---- annotation ----
    text(ax, 0.4, 0.10, sprintf('AMOC lower limb at %.1f°N', Lat_plot), ...
        'Units','normalized','FontName',fontname,'FontSize',fontsize+1, ...
        'Interpreter','none', 'Color',[0.15 0.15 0.15]);


    % ----- trend significance annotation -----
    isSig = corr_pval(Rho_ind,Lat_ind) > (1 - significant);

    if isSig
        t1 = trend_ci95(1, Rho_ind, Lat_ind);
        t2 = trend_ci95(2, Rho_ind, Lat_ind);
        text(ax, 0.5, 0.90, sprintf('-Trend = [%.2f, %.2f] Sv yr^{-1}', -t2, -t1), ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize-1, ...
            'Interpreter','tex', 'Color',[0.12 0.47 0.71]); %[0.85 0.2 0.2]
    else
        text(ax, 0.60, 0.90, 'Trend not significant', ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize-1, ...
            'Interpreter','none', 'Color',[0.35 0.35 0.35]);
    end



    % ---- export ----
    exportgraphics(gcf, fullfile(model_dir_baseline,'RealWorld','AA_AMOC26N_low.png'), 'Resolution',300);
    % exportgraphics(gcf, fullfile(model_dir_baseline,'RealWorld','AAAA5.pdf'), 'ContentType','vector');



    %% line plot - SO middepth MOC (-50.5)  [match AMOC(-20.5) style]
    Lat_plot = -50.5;

    figure('Position',framepos,'Color','w','Renderer','painters');
    ax = axes; hold(ax,'on');

    % ----- style -----
    fontname  = 'Arial';
    LWmean    = LW + 1.2;
    LWtrend   = LW + 0.6;
    alphaBand = 0.18;
    alphaGap  = 0.08;

    % ----- index -----
    lat_temp = Lat_plot;
    Lat_ind  = find(abs(lat_ACCESS-lat_temp) == min(abs(lat_ACCESS-lat_temp)),1);

    y = pred_strength_mid(:,Lat_ind);
    u = uncertainty_mid(:,Lat_ind);

    % ----- GRACE gap shading (put FIRST so it is behind everything) -----
    yl   = [14 34];
    xgap = [2017+7/12 2018+7/12];
    pgap = patch(ax, [xgap(1) xgap(2) xgap(2) xgap(1)], [yl(1) yl(1) yl(2) yl(2)], ...
        'cyan', 'EdgeColor','none', 'FaceAlpha', alphaGap);
    uistack(pgap,'bottom');

    % ----- uncertainty band (patch first) -----
    upper = y(:) + u(:);
    lower = y(:) - u(:);
    pband = patch(ax, [t_pred(:); flipud(t_pred(:))], [upper; flipud(lower)], ...
        Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);
    uistack(pband,'bottom');

    % ----- mean line -----
    plot(ax, t_pred, y, '-', 'LineWidth', LWmean, 'Color', Pred_color);

    % ----- trend line -----
    ri = DBNN_rho_ind_mid(Lat_ind);
    slope = trend_mean(ri, Lat_ind);
    b0    = intercepts_mean(ri, Lat_ind);
    trend_line = t_pred(:)*slope + b0;

    plot(ax, t_pred, trend_line, '--', 'LineWidth', LWtrend, 'Color', [0.85 0.2 0.2]);

    % ----- axes -----
    xlim(ax, t_range);
    ylim(ax, yl);
    xticks(ax, 2004:4:2024);

    xlabel(ax,'Year','FontName',fontname,'FontSize',fontsize,'Interpreter','none');
    ylabel(ax,'\Psi (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');

    ax.FontName   = fontname;
    ax.FontSize   = fontsize;
    ax.LineWidth  = LW;
    ax.TickDir    = 'out';
    ax.Box        = 'off';
    ax.Layer      = 'top';
    ax.TickLength = [0.012 0.012];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    % subtle grid (optional)
    ax.YGrid = 'on';
    ax.XGrid = 'off';
    ax.GridAlpha = 0.12;

    % ----- title-like annotation -----
    text(ax, 0.4, 0.10, sprintf('Mid-depth SMOC at %.1f°S', abs(Lat_plot)), ...
        'Units','normalized','FontName',fontname,'FontSize',fontsize+1, ...
        'Interpreter','none', 'Color',[0.1 0.1 0.1]);

    % ----- trend significance annotation (keep your test) -----
    isSig = corr_pval(ri,Lat_ind) > (1 - significant);

    if isSig
        t1 = trend_ci95(1, ri, Lat_ind);
        t2 = trend_ci95(2, ri, Lat_ind);
        text(ax, 0.02, 0.90, sprintf('Trend = [%.2f, %.2f] Sv yr^{-1}', t1, t2), ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize-1, ...
            'Interpreter','tex', 'Color',[0.85 0.2 0.2]);
    else
        text(ax, 0.60, 0.90, 'Trend not significant', ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize-1, ...
            'Interpreter','none', 'Color',[0.35 0.35 0.35]);
    end

    % export
    exportgraphics(gcf, fullfile(model_dir_baseline,'RealWorld','AA_SMOCm50S.png'), 'Resolution',300);
    % exportgraphics(gcf, fullfile(model_dir_baseline,'RealWorld','AAAA6.pdf'), 'ContentType','vector');


    %% line plot - SO abyssal MOC (-45.5)  [match AMOC(-20.5) style]
    Lat_plot = -45.5;

    figure('Position',framepos,'Color','w','Renderer','painters');
    ax = axes; hold(ax,'on');

    % ----- style -----
    fontname  = 'Arial';
    LWmean    = LW + 1.2;
    LWtrend   = LW + 0.6;
    alphaBand = 0.18;
    alphaGap  = 0.08;

    % ----- index -----
    lat_temp = Lat_plot;
    Lat_ind  = find(abs(lat_ACCESS-lat_temp) == min(abs(lat_ACCESS-lat_temp)),1);

    % Use sign convention you already applied (minus)
    y = -pred_strength_abys(:,Lat_ind);
    u =  uncertainty_abys(:,Lat_ind);

    % ----- GRACE gap shading (put FIRST so it is behind everything) -----
    yl   = [4 11];
    xgap = [2017+7/12 2018+7/12];
    pgap = patch(ax, [xgap(1) xgap(2) xgap(2) xgap(1)], [yl(1) yl(1) yl(2) yl(2)], ...
        'cyan', 'EdgeColor','none', 'FaceAlpha', alphaGap);
    uistack(pgap,'bottom');

    % ----- uncertainty band (patch first) -----
    upper = y(:) + u(:);
    lower = y(:) - u(:);
    pband = patch(ax, [t_pred(:); flipud(t_pred(:))], [upper; flipud(lower)], ...
        Pred_color, 'EdgeColor','none', 'FaceAlpha', alphaBand);
    uistack(pband,'bottom');

    % ----- mean line -----
    plot(ax, t_pred, y, '-', 'LineWidth', LWmean, 'Color', Pred_color);

    % ----- trend line (apply same sign convention) -----
    ri = DBNN_rho_ind_abys(Lat_ind);
    slope = trend_mean(ri, Lat_ind);
    b0    = intercepts_mean(ri, Lat_ind);
    trend_line = -(t_pred(:)*slope + b0);

    plot(ax, t_pred, trend_line, '--', 'LineWidth', LWtrend, 'Color', [0.12 0.47 0.71]);

    % ----- axes -----
    xlim(ax, t_range);
    ylim(ax, yl);
    xticks(ax, 2004:4:2024);

    xlabel(ax,'Year','FontName',fontname,'FontSize',fontsize,'Interpreter','none');
    ylabel(ax,'-\Psi (Sv)','FontName',fontname,'FontSize',fontsize,'Interpreter','tex');

    ax.FontName   = fontname;
    ax.FontSize   = fontsize;
    ax.LineWidth  = LW;
    ax.TickDir    = 'out';
    ax.Box        = 'off';
    ax.Layer      = 'top';
    ax.TickLength = [0.012 0.012];
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';

    % subtle grid (optional)
    ax.YGrid = 'on';
    ax.XGrid = 'off';
    ax.GridAlpha = 0.12;

    % ----- title-like annotation -----
    text(ax, 0.02, 0.10, sprintf('Abyssal SMOC at %.1f°S', abs(Lat_plot)), ...
        'Units','normalized','FontName',fontname,'FontSize',fontsize+1, ...
        'Interpreter','none', 'Color',[0.1 0.1 0.1]);

    % ----- trend significance annotation (keep your test) -----
    isSig = corr_pval(ri,Lat_ind) > (1 - significant);

    if isSig
        t1 = trend_ci95(1, ri, Lat_ind);
        t2 = trend_ci95(2, ri, Lat_ind);
        % CI should follow the sign convention shown in plot
        text(ax, 0.45, 0.90, sprintf('-Trend = [%.2f, %.2f] Sv yr^{-1}', -t2, -t1), ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize-1, ...
            'Interpreter','tex', 'Color',[0.12 0.47 0.71]);
    else
        text(ax, 0.60, 0.90, 'Trend not significant', ...
            'Units','normalized','FontName',fontname,'FontSize',fontsize-1, ...
            'Interpreter','none', 'Color',[0.35 0.35 0.35]);
    end

    % export
    exportgraphics(gcf, fullfile(model_dir_baseline,'RealWorld','AA_SMOCa45S.png'), 'Resolution',300);
    % exportgraphics(gcf, fullfile(model_dir_baseline,'RealWorld','AAAA8.pdf'), 'ContentType','vector');



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


function tf = useEccoV4r4(flag)
tf = logical(flag);
end
