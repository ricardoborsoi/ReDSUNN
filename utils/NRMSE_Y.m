% ===========================================================
function nrmse_val=NRMSE_Y(Y_true, Y_est)
% Y_true is a T-cell of size L,N
% Y_est is L,N,T or cell of T times L,N
    nrmse_val = 0;
    T = length(Y_true);
    [L,N] = size(Y_true{1});
    for t=1:T
        if iscell(Y_est)
            nrmse_val = nrmse_val + (1/T) * sum(sum(sum(...
                (Y_true{t} - Y_est{t}).^2))) ...
                / sum(sum(sum(Y_true{t}.^2)));
        else
            nrmse_val = nrmse_val + (1/T) * sum(sum(sum(...
                (Y_true{t} - Y_est(:,:,t)).^2))) ...
                / sum(sum(sum(Y_true{t}.^2)));
        end
    end
    nrmse_val = sqrt(nrmse_val);
end
