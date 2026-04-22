%INPUTS: read PIV field + define free-stream / reference pressures
dataTable = readtable('csv_out/PIV2D_alpha10.csv'); nx = max(dataTable.i); nz = max(dataTable.k);
Vinf = 14.5; P0 = 102335; rho = 1.206; q_inf = 0.5 * rho * Vinf^2; P_inf = P0 - q_inf;
X = reshape(dataTable.X, nx, nz); Z = reshape(dataTable.Z, nx, nz);
U = reshape(dataTable.U, nx, nz); W = reshape(dataTable.W, nx, nz);
dx = mean(diff(X(:,1))) / 1000; dz = mean(diff(Z(1,:))) / 1000;

%PREPROCESS: validity mask + Bernoulli pressure for IRR initialization
isValid = ~(U == 0 | W == 0); U(~isValid) = NaN; W(~isValid) = NaN;
speedSquared = U.^2 + W.^2; BernPressure = P0 - 0.5 * rho * speedSquared;

% BUILD ∇p : compute velocity gradients and form dp/dx, dp/dz
[Uz, Ux] = gradient(U, dz, dx); [Wz, Wx] = gradient(W, dz, dx);
Ux(isnan(Ux)) = 0; Uz(isnan(Uz)) = 0; Wx(isnan(Wx)) = 0; Wz(isnan(Wz)) = 0;
Px = -rho * (Ux .* U + Uz .* W); Pz = -rho * (Wz .* W + Wx .* U);   % ∂p/∂x and ∂p/∂z

% [Flowchart: Box 1] Initialize IRR^(0) and ROT^(0) by specifying DBC
xBC = -100; zBC = 50; dist2 = (X - xBC).^2 + (Z - zBC).^2; dist2(isnan(U)) = inf;
[~, linearIdxBC] = min(dist2(:)); [iBC, kBC] = ind2sub([nx nz], linearIdxBC);

% [Flowchart: Box 1] Initialize pressure field + partition (IRR=1, ROT=0, invalid=-1)
Pressure = zeros(nx, nz); Pressure(isnan(U)) = P0; Pressure(iBC, kBC) = BernPressure(iBC, kBC);
PartitionMask = zeros(nx, nz); PartitionMask(isnan(U)) = -1; PartitionMask(iBC, kBC) = 1;

%Neighbour stencil + stopping tolerances (δ_p, δ_c)
neighborDi = [-1  1  0  0 -1 -1  1  1]; neighborDk = [ 0  0  1 -1 -1  1 -1  1];
maxOuterIters = 40; cpTotLower = 0.985; cpTotUpper = 1.05;
deltaP = 1e-2; deltaC = 1e-2; cpPrev =[]; maskPrev =[]; %tolerances

for iter = 1:maxOuterIters

    % [Flowchart: Box 2] Perform spatial integration of ∇p in the estimated region
    %(Implementation: grow region by integrating from existing IRR neighbours)
    while any(PartitionMask(:) == 0)
        neighbors = conv2(double(PartitionMask == 1),ones(3),'same')-double(PartitionMask==1);
        neighbors(PartitionMask ~= 0) = -inf;
        bestNeighborCount = max(neighbors(:));
        [CandidateI, CandidateK] = find(neighbors == bestNeighborCount);
        for c = 1:numel(CandidateI)
            x0 = CandidateI(c); z0 = CandidateK(c);
            if PartitionMask(x0, z0) ~= 0, continue; end
            PressureSum = 0; numContrib = 0;

            % [Flowchart: Box 2] Local integration update from IRR neighbours using ∇p 
            for n = 1:8
                x = x0 + neighborDi(n); z = z0 + neighborDk(n);
                if x < 1 || x > nx || z < 1 || z > nz || PartitionMask(x, z) ~= 1, continue; 
                end
                dX = (X(x, z) - X(x0, z0)) / 1000; dZ = (Z(x, z) - Z(x0, z0)) / 1000; 
                ds = hypot(dX, dZ);
                if ds == 0, continue; end
                unitDirX = dX / ds; unitDirZ = dZ / ds;
                dp_ds = 0.5*((Px(x0, z0)+Px(x, z)) *unitDirX+(Pz(x0, z0)+Pz(x, z))*unitDirZ);
                PressureSum = PressureSum + (Pressure(x, z) - dp_ds * ds); 
                numContrib = numContrib + 1;
            end

            Pressure(x0, z0) = (numContrib > 0) * (PressureSum / numContrib) + ... 
            (numContrib == 0) * BernPressure(x0, z0);
            PartitionMask(x0, z0) = 2;
        end
        PartitionMask(PartitionMask == 2) = 1;
    end

    %Compute δ_p metric from successive Cp fields (Eq. 14)
    cpNow = (Pressure - P_inf)/q_inf; validCp = ~isnan(cpNow); relL2 = inf;
    if ~isempty(cpPrev)
        relL2 = norm(cpNow(validCp)-cpPrev(validCp),2) / max(norm(cpPrev(validCp),2), eps); 
    end
    cpPrev = cpNow;

    if iter ~= maxOuterIters
        Pressure(PartitionMask == -1) = P0;

        % Calculate Cp0 from current reconstructed pressure field (Eq. 3)
        cpTot = (Pressure - P_inf + 0.5 * rho * speedSquared) / q_inf;

        % [Flowchart: Box 3] Obtain new preliminary IRR–ROT partitioning from Cp0 thresholds
        PartitionMask = double(cpTot > cpTotLower & cpTot < cpTotUpper); 
        PartitionMask(isnan(U)) = -1;

        %Dilate/regularize interface (remove IRR cells touching ROT/invalid)
        touchesRotOrInvalid = conv2(double(PartitionMask <= 0),ones(3),'same') > 0;
        PartitionMask(PartitionMask > 0 & touchesRotOrInvalid) = 0;
        % [Flowchart: Box 4 & 5] Conservative assumption (for simplicity): 
        % all nodes in the uncertainty region are assigned to ROT. No integration is performed here.

        %Compute δ_c metric = fraction of nodes that flip class (Eq. 15)
        flipFrac = inf;
        if ~isempty(maskPrev)
            active = (PartitionMask~=-1); 
            flipFrac = nnz(xor(PartitionMask==1, maskPrev==1) & active) / nnz(active);
        end
        maskPrev = PartitionMask;

        % [Flowchart: Box 6] Terminate when both tolerances are satisfied 
        if (relL2 < deltaP) && (flipFrac < deltaC)
            fprintf('Converged at iter %d: relL2=%.3g, flipFrac=%.3g\n',iter,relL2,flipFrac); 
            break 
        end

        % PIV-based pressure field
        Pressure = zeros(nx, nz); Pressure(PartitionMask == -1) = P0;
        Pressure(PartitionMask == 1) = BernPressure(PartitionMask == 1);
    end
end

% [Flowchart: END] Final Cp from converged reconstructed pressure field
cp = (Pressure - P_inf) / q_inf; cp(PartitionMask == -1) = NaN;
figure; contourf(X, Z, cp, -2:0.25:1); clim([-2 1]); title('C_p', 'FontSize', 32); 
colormap(parula(12)); axis equal tight
