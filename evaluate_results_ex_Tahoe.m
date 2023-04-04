% =========================================================================
% Evaluate results for Lake Tahoe image
% 
% =========================================================================

clear
rng(1)
clc

% load dataset
load('../../BFHI/v1/data/rd_tip_with_outliers_t5.mat')
load('../../BFHI/v1/data/M1.mat')

% get constants
T = 6;
[L,P] = size(M0);
nr = H;
nc = W;
N = nr*nc;

% reorder image
Y_time = cell(T,1);
for t=1:T
    Y{t} = reshape(Y{t}',H,W,L);
    Y_time{t} = reshape(Y{t}, [N,L])';
end


%%
% load results from Python
load('saved/resultsVRNN_ex_config_Tahoe.mat')

A_hat_VRNN = permute(A_hat_VRNN, [2,1,3,4]);

% re-compute the reconstructed image
Y_hat_VRNN = permute(Y_hat_VRNN, [2,1,3]);
for t=1:T
    for n=1:N
        A_cube = reshape(A_hat_VRNN(:,:,:,t),[N,P]);
        Y_hat_VRNN(:,n,t) = Mn_hat_VRNN(:,:,n,t) * A_cube(n,:)';
    end
end
Y_hat_VRNN = permute(Y_hat_VRNN, [2,1,3]);


%%
% compute image resconstruction error and display times

RMSE_Y_propo  = NRMSE_Y(Y_time, permute(Y_hat_VRNN,[2,1,3]));

fprintf('ReDSUNN:\n')
fprintf('RMSE_Y...... %f\n\n', RMSE_Y_propo)
fprintf('TIMES...... %f\n\n', time_VRNN)


%% Plot images

fh = figure;
[ha, pos] = tight_subplot(1, T, 0.01, 0.1, 0.1);
for t=1:T
    Y_tmp = reshape(Y_time{t}',nr,nc,L);
    axes(ha(t));
    imagesc(3*Y_tmp(:,:,[32 20 8])), set(gca,'ytick',[],'xtick',[])
end






%% Plotting endmembers

fh = figure;
[ha, pos] = tight_subplot(1, P, 0.05, 0.1, 0.1);
for pp=1:P
    axes(ha(0*P + pp));
    plot(squeeze(Mn_hat_VRNN(:,pp,1,:)))
    ylim([0 0.5])
end

fh = figure;
[ha, pos] = tight_subplot(1, P, 0.05, 0.1, 0.1);
for pp=1:P
    axes(ha(0*T + pp));
    plot(squeeze(Mn_hat_VRNN(:,pp,1:200:end,3)))
    ylim([0 0.55])
end



%% Plotting abundances

fontSize = 12;


fh = figure;
[ha, pos] = tight_subplot(P, T, 0.01, 0.1, 0.1);
for pp=1:P
    for t=1:T
        A_cube = A_hat_VRNN(:,:,:,t);
        axes(ha((pp-1)*T + t));
        imagesc(A_cube(:,:,pp),[0 1]), set(gca,'ytick',[],'xtick',[])
    end
    colormap jet
end






