function C = mat_mul_mat(A, B, varargin)
% C = mat_mul_mat(A, B):           C = A*B
% C = mat_mul_mat(A, B, -1):       C = A'*B
% C = mat_mul_mat(A, B, 1, -1):    C = A*B'
% C = mat_mul_mat(A, B, -1, -1):   C = A'*B'
%
% Multiplication of N x 3 x 3 matrix A(:,3,3) and 3 x 3 matrix B(3,3)
%                   gives N x 3 x 3 matrix C(:,3)
%      or
% Multiplication of N x 3 x 3 matrices A(:,3,3) and B(:,3,3)
%                   gives N x 3 x 3 matrix C(:,3,3)
%      or
% Multiplication of N x 3 x 3 matrices A(:,3,3) and B(:,3)
%                   gives N x 3 matrix C(:,3)
%
% April 2005, Nils Olsen, DSRI

% Changes: NIO 25-04-2005: multiplication of transpose matrices

if ndims(A) ~= 3 ; error('ndims(A) <> 3'); end
if ndims(B) ~= 2 & ndims(B) ~= 3; error('ndims(B) neither 2 nor 3'); end

if nargin < 3
    itrans_A = 1;
    itrans_B = 1;
elseif nargin == 3
    itrans_A = varargin{1};
    itrans_B = 1;
elseif nargin == 4
    itrans_A = varargin{1};
    itrans_B = varargin{2};
else
    error('wrong number of arguments')
end

if (ndims(B) == 3 & size(A,1) == size(B,1)) | (ndims(B) == 2 & size(B,1) == 3)
    % multiplication with N x 3 x 3 (or 3 x 3) matrix B
    if itrans_A == 1 & itrans_B == 1
        if ndims(B) == 3
            C = zeros(size(A));
            for i=1:3
                for k=1:3
                    for l=1:3
                        C(:,i,k) = C(:,i,k) + A(:,i,l).*B(:,l,k);
                    end
                end
            end
        elseif ndims(B) == 2
            C = zeros(size(A));
            for i=1:3
                for k=1:3
                    for l=1:3
                        C(:,i,k) = C(:,i,k) + A(:,i,l).*B(l,k);
                    end
                end
            end
        end
    elseif itrans_A == -1 & itrans_B == 1
        if ndims(B) == 3
            C = zeros(size(A));
            for i=1:3
                for k=1:3
                    for l=1:3
                        C(:,i,k) = C(:,i,k) + A(:,l,i).*B(:,l,k);
                    end
                end
            end
        elseif ndims(B) == 2
            C = zeros(size(A));
            for i=1:3
                for k=1:3
                    for l=1:3
                        C(:,i,k) = C(:,i,k) + A(:,l,i).*B(l,k);
                    end
                end
            end
        end
    elseif itrans_A == -1 & itrans_B == -1
        if ndims(B) == 3
            C = zeros(size(A));
            for i=1:3
                for k=1:3
                    for l=1:3
                        C(:,i,k) = C(:,i,k) + A(:,l,i).*B(:,k,l);
                    end
                end
            end
        elseif ndims(B) == 2
            C = zeros(size(A));
            for i=1:3
                for k=1:3
                    for l=1:3
                        C(:,i,k) = C(:,i,k) + A(:,l,i).*B(k,l);
                    end
                end
            end
        end
    elseif itrans_A == 1 & itrans_B == -1
        if ndims(B) == 3
            C = zeros(size(A));
            for i=1:3
                for k=1:3
                    for l=1:3
                        C(:,i,k) = C(:,i,k) + A(:,i,l).*B(:,k,l);
                    end
                end
            end
        elseif ndims(B) == 2
            C = zeros(size(A));
            for i=1:3
                for k=1:3
                    for l=1:3
                        C(:,i,k) = C(:,i,k) + A(:,i,l).*B(k,l);
                    end
                end
            end
        end
    end
else
    % multiplication with N x 3 matrix B
    if itrans_A == 1
        C = zeros(size(B));
        for i=1:3
            for k=1:3
                C(:,i) = C(:,i) + A(:,i,k).*B(:,k);
            end
        end
    else
        C = zeros(size(B));
        for i=1:3
            for k=1:3
                C(:,i) = C(:,i) + A(:,k,i).*B(:,k);
            end
        end
    end
end