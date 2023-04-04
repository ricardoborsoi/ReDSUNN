% =========================================================================
% Evaluate results for synthetic example 2
% 
% =========================================================================

clear
rng(1)
addpath utils
clc

% load dataset
load ../../BFHI/baselines_VRNN/synth_dataset_ex2.mat

% get constants
[L,nr,nc,T] = size(Y);
P = size(M,2);
N = nr*nc;

% reorder image
Y_time = cell(T,1);
for t=1:T
    Y_time{t} = reshape(Y(:,:,:,t), [L,N]);
end


%%
% load results from Python
load('saved/resultsVRNN_ex_config_synth_ex2.mat')

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
% compute metrics and show stuff

RMSE_A_propo  = NRMSE_A(A, permute(reshape(A_hat_VRNN,[N,P,T]),[2,1,3]));
RMSE_M_propo  = NRMSE_M(M_nt, reshape(Mn_hat_VRNN,[L,P,nr,nc,T]));
SAM_M_propo   = SAM_M(M_nt, reshape(Mn_hat_VRNN,[L,P,nr,nc,T]));
RMSE_Y_propo  = NRMSE_Y(Y_time, permute(Y_hat_VRNN,[2,1,3]));

fprintf('ReDSUNN results:\n')
fprintf('RMSE_A...... %f\n', RMSE_A_propo)
fprintf('RMSE_M...... %f\n', RMSE_M_propo)
fprintf('SAM_M....... %f\n', SAM_M_propo)
fprintf('RMSE_Y...... %f\n', RMSE_Y_propo)
fprintf('TIME........ %f\n\n', time_VRNN)


%%
% fh = figure;
% [ha, pos] = tight_subplot(1, T, 0.01, 0.1, 0.1);
% for t=1:T
%     Y_cube = reshape(Y{t}',H,W,L);
%     axes(ha(t));
%     imagesc(3*Y_cube(:,:,[32 20 8])), set(gca,'ytick',[],'xtick',[])
% end



%% Plotting endmembers

fh = figure;
[ha, pos] = tight_subplot(1, P, 0.05, 0.1, 0.1);
for pp=1:P
    axes(ha(0*P + pp));
    plot(squeeze(Mn_hat_VRNN(:,pp,1,:)))
    ylim([0 0.35])
    xlim([1 173])
end

fh = figure;
[ha, pos] = tight_subplot(1, P, 0.05, 0.1, 0.1);
for pp=1:P
    axes(ha(0*T + pp));
    plot(squeeze(Mn_hat_VRNN(:,pp,1:200:end,3)))
    ylim([0 0.35])
    xlim([1 173])
end


%% Plotting abundances

fontSize = 12;

for pp=1:P
    fh = figure;
    [ha, pos] = tight_subplot(2, T, 0.01, 0.1, 0.1);
    for t=1:T
        A_cube = permute(A(:,:,:,t),[2,3,1]);
        axes(ha(0*T + t));
        imagesc(A_cube(:,:,pp),[0 1]), set(gca,'ytick',[],'xtick',[])
        
        A_cube = A_hat_VRNN(:,:,:,t);
        axes(ha(1*T + t));
        imagesc(A_cube(:,:,pp),[0 1]), set(gca,'ytick',[],'xtick',[])
    end
    axes(ha(0*T + 1)); ylabel('True','interpreter','latex','fontsize',fontSize)
    axes(ha(1*T + 1)); ylabel('ReDSUNN','interpreter','latex','fontsize',fontSize)
    colormap jet
    set(gcf,'PaperType','A3')
end










