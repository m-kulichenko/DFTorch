function[H,Hcoul,Hdipole,D,q,f,mu0,Dorth,SEnt] = ShadowPot(H0,S,Efield,Rx,Ry,Rz,H_Index_Start,H_Index_End,nocc,U,Znuc,Nats,Te,LBox,Element_Type,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);


% TO DO SIMPLIFIED SOLUTION q[n], and Kernel

N = max(size(H0));
it = 0; Res = 1;
Z = S^(-1/2);
h = sort(eig(Z'*H0*Z));
mu0 = 0.5*(h(nocc)+h(nocc+1)); 
%[D,mu0] = DM_Fermi(Z'*H0*Z,Te,mu0,nocc,20,1e-10,50);

[D,Ent,f,mu0] = DM_Fermi_x(Z'*H0*Z,Te,nocc,1e-10,50);

Nr_atoms = Nats;


D = Z*D*Z';
DS = 2*diag(D*S);
q = zeros(Nats,1);
for i = 1:Nats
  q(i) = sum(DS(H_Index_Start(i):H_Index_End(i))) - Znuc(i);
end


while Res > 1e-6
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

 [Dorth,SEnt,f,mu0] = DM_Fermi_x(Z'*H*Z,Te,nocc,1e-10,50);

 D = Z*Dorth*Z';
 q_old = q;

 DS = 2*diag(D*S);
 for i = 1:Nats
   q(i) = sum(DS(H_Index_Start(i):H_Index_End(i))) - Znuc(i);
 end

 Res = norm(q-q_old);

 q = q_old + 0.05*(q-q_old);

end
Dorth = 0.5*(Dorth+Dorth');

