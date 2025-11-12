%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Total energy calculation %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Etot,Eband0,Ecoul,S_ent] = ShadowEnergy(H0,U,D0,VCoul,D,q,n,Rx,Ry,Rz,f,Te);

%% Etot = 2*trace(H0*(D-D0)) + 0.5*sum_ij{i!=j} ((2qi-ni)*Cij*nj) + 0.5*sum_i (2qi-ni)*ni*Ui - Efield*mu - 2*Te*S_ent
%% dipole = mu(:) = sum_i qi*R(i,:); qi = 2*(D_ii-D0_ii)

HDIM = max(size(H0));

kB = 8.61739e-5; % eV/K;
N = max(size(q));
D0 = diag(D0);
Eband0 = 2*trace(H0*(D-D0));  % Single-particle/band energy

Ecoul = 0.5*(2*q-n)'*VCoul';           % Coulomb energy
for i = 1:N
  Ecoul = Ecoul + 0.5*(2.0*q(i)-n(i))*U(i)*n(i);
end

S_ent = 0.0; eps = 1e-14;
for i = 1:HDIM
  if (f(i) < 1-eps) & (f(i) > eps)
    S_ent = S_ent - kB*(f(i)*log(f(i)) + (1-f(i))*log(1-f(i)));
  end
end

Etot = Eband0 + Ecoul - 2*Te*S_ent;         % Total energy
