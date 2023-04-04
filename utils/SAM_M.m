function sam_val = SAM_M(M_true, M_est)
% M_true (L,P,nr,nc,T) 224     3    50    50     6
% M_est : either L * P * T, or (L,P,nr,nc,T) 224     3    50    50     6

[L,P,nr,nc,T] = size(M_true);

if ~iscell(M_est) && length(size(M_est)) < 4
    tmp = M_est;
    M_est = cell(T,1);
    for t=1:T
        M_est{t} = tmp(:,:,t);
    end
end


if iscell(M_est)
    sam_val = 0;
    for i=1:nr
        for j=1:nc
            for t=1:T
                for p=1:P
                    sam_val = sam_val + (1/(T*nr*nc*P)) * acos( ...
                        (M_true(:,p,i,j,t)'*M_est{t}(:,p)) ...
                        / ( norm(M_true(:,p,i,j,t))*norm(M_est{t}(:,p)) ) );
                end
            end
        end
    end
    
    
else
    sam_val = 0;
    for i=1:nr
        for j=1:nc
            for t=1:T
                for p=1:P
                    sam_val = sam_val + (1/(T*nr*nc*P)) * acos( ...
                        (M_true(:,p,i,j,t)'*M_est(:,p,i,j,t)) ...
                        / ( norm(M_true(:,p,i,j,t))*norm(M_est(:,p,i,j,t)) ) );
                end
            end
        end
    end
    
end

