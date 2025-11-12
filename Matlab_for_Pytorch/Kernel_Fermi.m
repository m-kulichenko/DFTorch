function [KK,D0,mu0] = Kernel_Fermi(mu0,T,RX,RY,RZ,LBox,Hubbard_U,Element_Type,Nr_atoms,HDIM,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType,H_INDEX_START,H_INDEX_END,H,S,Z,Q,e,Nocc,Znuc)

  dq_dn = zeros(Nr_atoms);
  dq_J = zeros(1,Nr_atoms);
  for J = 1:Nr_atoms
    dq_J(J) = 1;
    for I = 1:Nr_atoms
      [Coulomb_Pot_Real(I),Coulomb_Force_Real(:,I)] = Ewald_Real_Space(I,RX,RY,RZ,LBox,dq_J,Hubbard_U,Element_Type,Nr_atoms,Coulomb_acc,TIMERATIO,nnRx,nnRy,nnRz,nrnnlist,nnType);
    end
    [Coulomb_Pot_k,Coulomb_Force_k] = Ewald_k_Space(RX,RY,RZ,LBox,dq_J,Nr_atoms,Coulomb_acc,TIMERATIO);
    Coulomb_Pot_dq_J = Coulomb_Pot_Real+Coulomb_Pot_k;

    H_dq_J = eye(HDIM);
    for I = 1:Nr_atoms
       for K = H_INDEX_START(I):H_INDEX_END(I)
           H_dq_J(K,K) = Hubbard_U(I)*dq_J(I) + Coulomb_Pot_dq_J(I);
       end
    end
    H_dq_J =  0.5*S*H_dq_J + 0.5*H_dq_J*S;

    MaxIt = 6; m = 18; eps = 1e-6;
    H0 = Z'*H*Z;
    H1 = Z'*H_dq_J*Z; mu1 = 0;
%    [D0,D_dq_J,mu0,mu1] = Rec_Fermi_PRT1(H0,H1,T,mu0,mu1,Nocc,m,eps,MaxIt);

%[P0,P1,mu0,mu1] = Rec_Fermi_PRT1(H0,H1,Te,mu0,mu1,Ne,m,eps,MaxIt);
    [D0,D_dq_J] = Fermi_PRT(H0,H1,T,Q,e,mu0);

    D_dq_J = 2*Z*D_dq_J*Z';
    D_diag = diag(D_dq_J*S);
    for i = 1:Nr_atoms
      dqI_dqJ(i) = sum(D_diag(H_INDEX_START(i):H_INDEX_END(i))); % Net Charge
    end
    dq_dn(J,:) = dqI_dqJ(:);
    dq_J(J) = 0;
  end
  II = eye(Nr_atoms);
  KK = (dq_dn'-II)^(-1);
