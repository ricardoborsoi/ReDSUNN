% ===========================================================
function nrmse_val=NRMSE_A(A_true, A_est)
% A_true is of size P,nr,nc,T
% A_est is P,N,T or cell of T times P,N
    nrmse_val = 0;
    %T = size(A_true,4);
    [P,nr,nc,T] = size(A_true);
    N = nr*nc;
    for t=1:T
        if iscell(A_est)
            nrmse_val = nrmse_val + (1/T) * sum(sum(sum(...
                (reshape(A_true(:,:,:,t),[P,N]) - A_est{t}).^2))) ...
                / sum(sum(sum(A_true(:,:,:,t).^2)));
        else
            nrmse_val = nrmse_val + (1/T) * sum(sum(sum(...
                (reshape(A_true(:,:,:,t),[P,N]) - A_est(:,:,t)).^2))) ...
                / sum(sum(sum(A_true(:,:,:,t).^2)));
        end
    end
    nrmse_val = sqrt(nrmse_val);
end
