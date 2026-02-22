%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  buildNeighborList.m
%
%  O(N) cell-linked list neighbor search with periodic boundary
%  conditions (orthorhombic box)
%
%  Inputs:
%    R     : N×3 array of atom coordinates
%    Lbox  : [Lx, Ly, Lz] box dimensions
%    Rcut  : cutoff distance
%
%  Output:
%    nbrlist : cell array where nbrlist{i} contains neighbor indices of atom i
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nbrlist = buildNeighborList(R, Lbox, Rcut)

N = size(R,1);
Lx = Lbox(1);  Ly = Lbox(2);  Lz = Lbox(3);

% --- number of cells in each direction ---
nx = max(1, floor(Lx / Rcut));
ny = max(1, floor(Ly / Rcut));
nz = max(1, floor(Lz / Rcut));
cellsize = [Lx/nx, Ly/ny, Lz/nz];

ncell = nx * ny * nz;

% --- assign atoms to cells ---
cellAtoms = cell(ncell,1);
for i = 1:N
    % map coordinates to cell indices (0-based wrapped)
    ix = mod(floor(R(i,1)/cellsize(1)), nx);
    iy = mod(floor(R(i,2)/cellsize(2)), ny);
    iz = mod(floor(R(i,3)/cellsize(3)), nz);
    cid = 1 + ix + nx*(iy + ny*iz);
    cellAtoms{cid}(end+1) = i; %#ok<AGROW>
end

% --- helper: minimum-image distance (scalar form) ---
minimg = @(dx,L) dx - L*round(dx/L);

% --- allocate neighbor list ---
nbrlist = cell(N,1);

% --- search loop ---
for iz = 0:nz-1
  for iy = 0:ny-1
    for ix = 0:nx-1
      cid = 1 + ix + nx*(iy + ny*iz);
      atoms1 = cellAtoms{cid};
      if isempty(atoms1), continue; end

      for a = 1:numel(atoms1)
        i = atoms1(a);
        xi = R(i,1); yi = R(i,2); zi = R(i,3);

        % loop over this and 26 neighboring cells
        for dz = -1:1
          for dy = -1:1
            for dx = -1:1
              jx = mod(ix + dx, nx);
              jy = mod(iy + dy, ny);
              jz = mod(iz + dz, nz);
              cid2 = 1 + jx + nx*(jy + ny*jz);
              atoms2 = cellAtoms{cid2};
              if isempty(atoms2), continue; end

              for b = 1:numel(atoms2)
                j = atoms2(b);
                if j <= i, continue; end  % avoid double counting

                dx_ = minimg(R(j,1) - xi, Lx);
                dy_ = minimg(R(j,2) - yi, Ly);
                dz_ = minimg(R(j,3) - zi, Lz);
                dist2 = dx_*dx_ + dy_*dy_ + dz_*dz_;
                if dist2 < Rcut^2
                    % record both directions (symmetric)
                    nbrlist{i}(end+1) = j; %#ok<AGROW>
                    nbrlist{j}(end+1) = i; %#ok<AGROW>
                end
              end
            end
          end
        end
      end
    end
  end
end

end
