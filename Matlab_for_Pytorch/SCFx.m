function[H,Hcoul,Hdipole,KK,D,q,f,mu0,Dorth,SEnt] = SCFx(H0,S,Efield,Rx,Ry,Rz,H_Index_Start,H_Index_End,nocc,U,Znuc,Nats,Te,LBox,Element_Type,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);


m = 3; % Kernel calculation
FelTol = 1e-16;
MaxRank = 10;
Fel = 100.0;
N = max(size(H0));
it = 0; ResNorm = 1;
Z = S^(-1/2);
h = sort(eig(Z'*H0*Z));
mu0 = 0.5*(h(nocc)+h(nocc+1));
%[D,mu0] = DM_Fermi(Z'*H0*Z,Te,mu0,nocc,20,1e-10,50);
H0_orth = Z'*H0*Z;

[D,Ent,Q,e,f,mu0] = DM_Fermi_x(H0_orth,Te,nocc,1e-10,50);

Nr_atoms = Nats;


D = Z*D*Z';
DS = 2*diag(D*S);
q = zeros(Nats,1);
for i = 1:Nats
  q(i) = sum(DS(H_Index_Start(i):H_Index_End(i))) - Znuc(i);
end

KK = -0.5;  % Initial mixing coefficient for linear mixing
KK0 = KK*eye(Nats);

while (ResNorm > 1e-6)
 it = it + 1;
 Dipole = diag(-Rx*Efield(1)-Ry*Efield(2)-Rz*Efield(3));
 Hdipole = zeros(N);

 for I = 1:Nr_atoms
    [Coulomb_Pot_Real(I),Coulomb_Force_Real(:,I)] = Ewald_Real_Space(I,Rx,Ry,Rz,LBox,q',U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
 end
 [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(Rx,Ry,Rz,LBox,q',Nr_atoms,Coulomb_acc,TIMERATIO);
 Coulomb_Pot = Coulomb_Pot_Real + Coulomb_Pot_k;

 CoulPot = Coulomb_Pot;

 Hcoul = zeros(N);
 for i = 1:Nats
   for j = H_Index_Start(i):H_Index_End(i)
     Hdipole(j,j) = Dipole(i);
     Hcoul(j,j) = U(i)*q(i) + CoulPot(i);
   end
 end

 Hcoul = 0.5*Hcoul*S + 0.5*S*Hcoul;
 Hdipole = 0.5*Hdipole*S + 0.5*S*Hdipole;
 H = H0 + Hcoul + Hdipole;

 [Dorth,SEnt,Q,e,f,mu0] = DM_Fermi_x(Z'*H*Z,Te,nocc,1e-10,50);

 if (it == m) % Calculate full kernel after m steps
   [KK,D0,mu_0] = Kernel_Fermi(mu0,Te,Rx,Ry,Rz,LBox,U,Element_Type,Nr_atoms,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_Index_Start,H_Index_End,H,S,Z,Q,e,nocc,Znuc) ;
   KK0 = KK;  % To be kept as preconditioner
   %save("KK.mat","KK",'-v4')
   %error('MyTool:Fatal','Something went wrong. Stopping.');


 end

 D = Z*Dorth*Z';
 q_old = q;

 DS = 2*diag(D*S);
 for i = 1:Nats
   q(i) = sum(DS(H_Index_Start(i):H_Index_End(i))) - Znuc(i);
 end

 Res = (q-q_old);
 ResNorm = norm(Res);
 Initial_SCF_Error_Res_Norm = ResNorm
 pause
 K0Res = KK*Res;

 % Preconditioned Low-Rank Krylov SCF acceleration
 vi = zeros(Nats,MaxRank);
 fi = zeros(Nats,MaxRank);
 v = zeros(Nats,1);
 if (it > m)
    K0Res = KK0*Res;
    dr = K0Res;
    I = 0;
    Fel = 100.0;
    while ((I < MaxRank) & (Fel > FelTol))
       I = I + 1;
       vi(:,I) = dr/norm(dr);
       for J = 1:I-1
          vi(:,I) = vi(:,I) - (vi(:,I)'*vi(:,J))*vi(:,J);  %!! Orthogonalized v_i as in Eq. (42) Ref. [*]
       end
       vi(:,I) = vi(:,I)/norm(vi(:,I));
       v(:) = vi(:,I);  % v_i
       for II = 1:Nr_atoms
          [Coulomb_Pot_Real(II),Coulomb_Force_Real(:,II)] = Ewald_Real_Space(II,Rx,Ry,Rz,LBox,v,U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
       end
       [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(Rx,Ry,Rz,LBox,v',Nr_atoms,Coulomb_acc,TIMERATIO);
       dCoulPot = Coulomb_Pot_Real + Coulomb_Pot_k;

       H1 = zeros(N);
       for i = 1:Nats
         for j = H_Index_Start(i):H_Index_End(i)
           H1(j,j) = U(i)*v(i) + dCoulPot(i);
         end
       end

       H1 = 0.5*H1*S + 0.5*S*H1;
       H1_orth = Z'*H1*Z;
       [D0,D1] = Fermi_PRT(H0_orth, H1_orth, Te, Q, e, mu0);
       D1 = Z*D1*Z';
       D1S = 2*diag(D1*S);
       for i = 1:Nats
         dq(i) = sum(D1S(H_Index_Start(i):H_Index_End(i)));
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
       OI
       pause
       KRes = zeros(Nats,1);

       fi(:,1:I)'*K0Res
       pause

       Y = OI*fi(:,1:I)'*K0Res;
       Y
       pause
       fi(:,1:I)*Y
       pause

       Fel = norm(fi(:,1:I)*Y - K0Res);
    end
    K0Res = vi(:,1:I)*Y;
 end

 % Mixing
 q = q_old - K0Res;

end
Dorth = 0.5*(Dorth+Dorth');

