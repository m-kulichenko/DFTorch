%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  SEDACS PROXY MATLAB VERSION                      %
%                  SHADOW XL-BOMD for SCC-DFTB                      %
%                   A.M.N. Niklasson, T1, LANL                      %
% i                        (OCT 13 2025)                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Total Energy Function:                                            %
% E = 2Tr[H0(D-D0)] + (1/2)*sum_i U_i q_i^2 +                       %
%      + (1/2)sum_{i,j (i!=j)} q_i C_{ij} q_j - Efield*dipole*0     %
% dipole = sum_i R_{i} q_i                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To Do: -> dSx, dHx etc analytically
% To Do: -> U to Us, Up, Ud, etc
% To Do: -> Atomic multipoles
% To Do: -> Non-periodic BC  (for training)
clear;

% Initial data, load atoms and coordinates, etc in COORD.dat
Nats = 24;  % Number of atoms
Nr_atoms = Nats;
Nocc = 32;  % Nr of electrons / 2
Efield = 0.0*0.3*[-.3,0.4,0.0]'; % In arbitrary direction  Works ony in 0-field!!!  FIELDS ARE IGNORED NO FIELD! NOT WORKED OUT FOR FIELDS
Te = 3000;                    % Some electronic temperature in Kelvin, Possible bug at high tempertures!!!
A = importdata('COORD_8WATER.dat');   % Input coordinate file
TYPE = A.textdata(:);
RX = A.data(:,1); RY = A.data(:,2); RZ = A.data(:,3); % Atomic coordinates
LBox(1) = 10; LBox(2) = 10; LBox(3) = 10;              % PBC Box dimensions
% LBox(1) = 12; LBox(2) = 12; LBox(3) = 12;              % PBC Box dimensions
% LBox(1) = 14.404799; LBox(2) = 14.404799; LBox(3) = 14.404799;              % PBC Box dimensions

% Load Pair Potential Parameters
[PairPotType1,PairPotType2,PotCoef] = LoadPairPotParameters;

% Get Hamiltonian, Overlap, atomic DM = D0 (vector only), etc, but first the neighborlist
[nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,4.0,Nats);
[H0,S,D0,H_INDEX_START,H_INDEX_END,Element_Type,Mnuc,Znuc,Hubbard_U] = H0_and_S(TYPE,RX,RY,RZ,LBox,Nats,nrnnlist,nnRx,nnRy,nnRz,nnType);
HDIM = max(size(H0));           % Total number of basis functions
Z = S^(-1/2); Z0 = Z; S0 = S;   % Factorization Z of the inverse overlap matrix S
Rcut = 10.42; Coulomb_acc = 1e-8; TIMERATIO = 10; % Parameters for the Coulomb summations
F2V = 0.01602176487/1.660548782;
MVV2KE = 166.0538782/1.602176487;
KE2T = 1/0.000086173435;

% Get full Coulomb matrix. In principle we do not need an explicit representation of the Coulomb matrix C!
[nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,12.0,Nats);
C = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nats,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);

% SCF ground state optimization for H and D and q and occupation factors f, D*S*D = D, Tr[DS] = Nocc, f in [0,1]
[H,Hcoul,Hdipole,KK,D,q,f,mu0,Dorth,SEnt] = SCFx(H0,S,Efield,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Nocc,Hubbard_U,Znuc,Nats,Te,LBox,Element_Type,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);

% Get the energy and the partial energies
[Etot,Eband0,Ecoul,Edipole,S_ent] = Energy(H0,Hubbard_U,Efield,D0,C,D,q,RX,RY,RZ,f,Te); % Energy calculation - 2*Te*S_ent
[PairForces,ERep] = PairPotentialForces(RX,RY,RZ,LBox,Element_Type,Nats,PairPotType1,PairPotType2,PotCoef);

EPOT = Eband0 + Ecoul + Edipole + ERep - 2*Te*S_ent;

ZI = S^(1/2);
DO = ZI*D*ZI';

dx = 0.0001; % Calculate derivatives for S, H and C using simple finite differences
[nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,4.0,Nats);
[dSx,dSy,dSz] = GetdS(Nats,dx,HDIM,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox,nrnnlist,nnRx,nnRy,nnRz,nnType);
[dHx,dHy,dHz] = GetdH(Nats,dx,HDIM,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox,nrnnlist,nnRx,nnRy,nnRz,nnType);
[dCx,dCy,dCz] = GetdC(Nats,dx,Coulomb_acc,Rcut,TIMERATIO,HDIM,Hubbard_U,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox);

% Get the forces and the partial forces
[Ftot,Fcoul,Fband0,Fdipole,FPulay,FScoul,FSdipole] = Forces(H,H0,S,C,D,D0,dHx,dHy,dHz,dSx,dSy,dSz,dCx,dCy,dCz,Efield,Hubbard_U,q,RX,RY,RZ,Nats,H_INDEX_START,H_INDEX_END);
Ftot = Ftot + PairForces;

% Initial BC for n
n = q; n_0 = q; n_1 = q; n_2 = q; n_3 = q; n_4 = q; n_5 = q;
mu_0 = mu0; mu_1 = mu0; mu_2 = mu0; mu_3 = mu0; mu_4 = mu0; mu_5 = mu0;
C0 = -6; C1 = 14; C2 = -8; C3 = -3; C4 = 4; C5 = -1; % Coefficients for modified Verlet integration
kappa = 1.82; alpha = 0.018;                         % Coefficients for modified Verlet integration
dt = 0.25;                              % Time step in fs
VX = 0*RX; VY = 0*RX; VZ = 0*RX;        % Initialize velocities
KK0 = KK;
K0Res = KK*(q-n);

for MD_step = 1:1000  %% MAIN MD LOOP

  %% OUTPUTS FOR SHADOW MD SIMULATIONS
  EKIN = 0.5*MVV2KE*sum(Mnuc'.*(VX.^2+VY.^2+VZ.^2));            % Kinetic energy in eV (MVV2KE: unit conversion)
  Temperature(MD_step) = (2/3)*KE2T*EKIN/Nr_atoms;              % Statistical temperature in Kelvin
  Energ(MD_step) = EKIN + EPOT;                                % Total Energy in eV, Total energy fluctuations Propto dt^2
  Time(MD_step) = MD_step*dt;
  ResErr(MD_step) = norm(q-n)/sqrt(Nats);                      % ResErr Propto dt^2
  [Time(MD_step),Energ(MD_step)/1000,Temperature(MD_step),ResErr(MD_step)]

  VX = VX + 0.5*dt*(F2V*Ftot(1,:)./Mnuc)' - 0.0*VX;      % First 1/2 of Leapfrog step
  VY = VY + 0.5*dt*(F2V*Ftot(2,:)./Mnuc)' - 0.0*VY;      % F2V: Unit conversion
  VZ = VZ + 0.5*dt*(F2V*Ftot(3,:)./Mnuc)' - 0.0*VZ;      % -c*V c>0 => Fricition

  RX = RX + dt*VX;                              % Update positions
  RY = RY + dt*VY;
  RZ = RZ + dt*VZ;

  % Get full Coulomb matrix. In principle we do not need an explicit representation of the Coulomb matrix C!
%  [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,Rcut,Nats);
%  C = CoulombMatrix(RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nats,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END);

  [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,4.0,Nats);
  [H0,S,D0,H_INDEX_START,H_INDEX_END,Element_Type,Mnuc,Znuc,Hubbard_U] = H0_and_S(TYPE,RX,RY,RZ,LBox,Nats,nrnnlist,nnRx,nnRy,nnRz,nnType);
  HDIM = max(size(H0));           % Total number of basis functions
  Z = S^(-1/2); Z0 = Z; S0 = S;   % Factorization Z of the inverse overlap matrix S
  n = 2*n_0 - n_1 - kappa*K0Res + alpha*(C0*n_0+C1*n_1+C2*n_2+C3*n_3+C4*n_4+C5*n_5);
  %n = 2*n_0 - n_1 - kappa*KK*(q-n) + alpha*(C0*n_0+C1*n_1+C2*n_2+C3*n_3+C4*n_4+C5*n_5);
  %n = 2*n_0 - n_1 + 0.4*kappa*(q-n) + alpha*(C0*n_0+C1*n_1+C2*n_2+C3*n_3+C4*n_4+C5*n_5);
  n_5 = n_4; n_4 = n_3; n_3 = n_2; n_2 = n_1; n_1 = n_0; n_0 = n;

 [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,12.0,Nats);
 Coulomb_Pot_Real = zeros(1,Nats);
 Coulomb_Pot_k = zeros(1,Nats);
 for I = 1:Nr_atoms
    [Coulomb_Pot_Real(I),Coulomb_Force_Real(:,I)] = Ewald_Real_Space(I,RX,RY,RZ,LBox,n,Hubbard_U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
 end
 [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(RX,RY,RZ,LBox,n',Nr_atoms,Coulomb_acc,TIMERATIO);
 CoulPot = Coulomb_Pot_Real + Coulomb_Pot_k;
 FCoul = Coulomb_Force_Real + Coulomb_Force_k;

 Hcoul = zeros(HDIM);
 for i = 1:Nats
   for j = H_INDEX_START(i):H_INDEX_END(i)
     Hcoul(j,j) = Hubbard_U(i)*n(i) + CoulPot(i);
   end
 end

 H = H0 + 0.5*Hcoul*S + 0.5*S*Hcoul;

 H_orth = Z'*H*Z;
 [Dorth,SEnt,Q,e,f,mu0] = DM_Fermi_x(H_orth,Te,Nocc,1e-10,50);

 D = Z*Dorth*Z';
 DS = 2*diag(D*S);
 for i = 1:Nats
   q(i) = sum(DS(H_INDEX_START(i):H_INDEX_END(i))) - Znuc(i);
 end

%% Update Kernel %%
  NoRank = 0;
  Res = q - n;
  if (mod(MD_step,1000) == 1)
     [KK,D_0,mu_0] = Kernel_Fermi(mu0,Te,RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nr_atoms,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END,H,S,Z,Q,e,Nocc,Znuc);
      KK0 = KK;
      K0Res = KK0*Res;
  elseif (NoRank == 1)
       K0Res = KK0*Res;
  else
% Preconditioned Low-Rank Krylov SCF acceleration
     Fel = 100.0; MaxRank = 10; FelTol = 1e-4;
     vi = zeros(Nats,MaxRank);
     fi = zeros(Nats,MaxRank);
     v = zeros(Nats,1);
     K0Res = KK0*Res;
     dr = K0Res;
     I = 0; % Rank Counter
     while ((I < MaxRank) & (Fel > FelTol))
        I = I + 1;
        vi(:,I) = dr/norm(dr);
        for J = 1:I-1
           vi(:,I) = vi(:,I) - (vi(:,I)'*vi(:,J))*vi(:,J);  %!! Orthogonalized v_i as in Eq. (42) Ref. [*]
        end
        vi(:,I) = vi(:,I)/norm(vi(:,I));
        v(:) = vi(:,I);  % v_i
        for II = 1:Nr_atoms
           [Coulomb_Pot_Real(II),Coulomb_Force_Real(:,II)] = Ewald_Real_Space(II,RX,RY,RZ,LBox,v,Hubbard_U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
        end
        [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(RX,RY,RZ,LBox,v',Nr_atoms,Coulomb_acc,TIMERATIO);
        dCoulPot = Coulomb_Pot_Real + Coulomb_Pot_k;

        H1 = zeros(HDIM);
        for i = 1:Nats
          for j = H_INDEX_START(i):H_INDEX_END(i)
            H1(j,j) = Hubbard_U(i)*v(i) + dCoulPot(i);
          end
        end

        H1 = 0.5*H1*S + 0.5*S*H1;
        H1_orth = Z'*H1*Z;
        [DD0,D1] = Fermi_PRT(H_orth, H1_orth, Te, Q, e, mu0);
        D1 = Z*D1*Z';
        D1S = 2*diag(D1*S);
        for i = 1:Nats
          dq(i) = sum(D1S(H_INDEX_START(i):H_INDEX_END(i)));
        end
        dr = dq' - v;      %!! dr = df/dlambda, last row in Eq. (42) Ref[*]
        dr = KK0*dr;
        fi(:,I) = dr;  % ! fv_i
        O = zeros(I);
        for k = 1:I
        for l = 1:I
            O(k,l) = fi(:,k)'*fi(:,l);
        end
        end
        OI = O^(-1);
        KRes = zeros(Nats,1);
        Y = OI*fi(:,1:I)'*K0Res;
        Fel = norm(fi(:,1:I)*Y - K0Res)/sqrt(Nats*1.0);
     end
     K0Res = vi(:,1:I)*Y;
  end

  % Get the energy and the partial energies
  [Etot,Eband0,Ecoul,S_ent] = ShadowEnergy(H0,Hubbard_U,D0,CoulPot,D,q,n,RX,RY,RZ,f,Te);
  [PairForces,ERep] = PairPotentialForces(RX,RY,RZ,LBox,Element_Type,Nats,PairPotType1,PairPotType2,PotCoef);

  EPOT = Eband0 + Ecoul + Edipole + ERep - 2*Te*S_ent;

  dx = 0.0001; % Calculate derivatives for S, H and C using simple finite differences
  [nrnnlist,nndist,nnRx,nnRy,nnRz,nnType,nnStruct,nrnnStruct]  = nearestneighborlist(RX,RY,RZ,LBox,4.0,Nats);
  [dSx,dSy,dSz] = GetdS(Nats,dx,HDIM,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox,nrnnlist,nnRx,nnRy,nnRz,nnType);
  [dHx,dHy,dHz] = GetdH(Nats,dx,HDIM,RX,RY,RZ,H_INDEX_START,H_INDEX_END,Element_Type,LBox,nrnnlist,nnRx,nnRy,nnRz,nnType);

  % Get the forces and the partial forces

  for i = 1:Nats
    FCoul(1,i) = (2*q(i)-n(i))*FCoul(1,i)/n(i);
    FCoul(2,i) = (2*q(i)-n(i))*FCoul(2,i)/n(i);
    FCoul(3,i) = (2*q(i)-n(i))*FCoul(3,i)/n(i);
  end

  [Ftot,Fband0,Fdipole,FPulay,FScoul,FSdipole] = ShadowForces(H,H0,S,D,D0,dHx,dHy,dHz,dSx,dSy,dSz,Efield,Hubbard_U,q,n,CoulPot,RX,RY,RZ,Nats,H_INDEX_START,H_INDEX_END);
  Ftot = Ftot + FCoul + PairForces;

  VX = VX + 0.5*dt*(F2V*Ftot(1,:)./Mnuc)' - 0.0*VX;      % Integrate second 1/2 of leapfrog step
  VY = VY + 0.5*dt*(F2V*Ftot(2,:)./Mnuc)' - 0.0*VY;      % - c*V  c > 0 => friction
  VZ = VZ + 0.5*dt*(F2V*Ftot(3,:)./Mnuc)' - 0.0*VZ;

  for i = 1:Nr_atoms
    RX(i) = mod(RX(i)*10000000,10000000*LBox(1))/10000000;
    RY(i) = mod(RY(i)*10000000,10000000*LBox(2))/10000000;
    RZ(i) = mod(RZ(i)*10000000,10000000*LBox(3))/10000000;
  end

end
