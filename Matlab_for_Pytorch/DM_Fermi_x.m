  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function [P0,S,Q,e,f,mu0] = DM_Fermi_x(H0,T,Nocc,eps,MaxIt)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  N = max(size(H0));
  I = eye(N); 
  OccErr = 1;
  Cnt = 0;
  [Q,E] = eig(H0);
  e = diag(E); h = sort(e);
  iter = 0;
  mu0 = 0.5*(h(Nocc) + h(Nocc+1)); 
  %kB = 6.33366256e-6; % Ry/K;
  %kB = 3.166811429e-6; % Ha/K;
  kB = 8.61739e-5; % eV/K;
  beta = 1.0/(kB*T);   % Temp in Kelvin
  while abs(OccErr) > eps
  iter = iter + 1;
     f = 1./(exp(beta*(e-mu0))+1);
     dOcc = beta*sum(f.*(1.0-f));
     Occ = sum(f);
     OccErr = Nocc-Occ;
     if abs(OccErr) > 1e-10
        mu0 = mu0 + OccErr/dOcc;
     end
     if iter > MaxIt
        OccErr = 0.0;
        NonConv = 1
        pause 
     end
  end
  P0 = Q*diag(f)*Q';
  S = 0.0;
  HDIM = max(size(H0));
  for i = 1:HDIM
   p_i = f(i);
   if ((p_i > 1e-14) & ((1.D0-p_i) > 1e-14)) 
     S = S - kB*(p_i*log(p_i) + (1.D0-p_i)*log(1.D0-p_i));
   end
 end

