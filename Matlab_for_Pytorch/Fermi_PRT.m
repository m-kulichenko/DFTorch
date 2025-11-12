function [D0,D1] = Fermi_PRT(H0, H1, Te, Q, ev, mu0)
kB = 8.61739e-5; % eV/K;
beta = 1/(kB*Te);

N  = max(size(H0));

QH1Q = Q'*H1*Q;
fe = 1./(exp(beta*(ev-mu0))+1);
dDtmp = zeros(N);
X = zeros(N);
for i = 1:N
  for j = 1:N
    if abs(ev(i)-ev(j)) < 1e-4
      xx = (ev(i)+ev(j))/2;
      tmp = beta*(xx-mu0);
      if abs(tmp) > 25
        dDtmp(i,j) = 0.0;
      else
        dDtmp(i,j) = -beta*exp(beta*(xx-mu0))/(exp(beta*(xx-mu0))+1)^2;
      end
    else
      dDtmp(i,j) = (fe(i)-fe(j))/(ev(i)-ev(j));
    end
    X(i,j) = dDtmp(i,j)*QH1Q(i,j);
  end
end
TrdD = trace(dDtmp);
if abs(TrdD) > 10e-9
  mu1 = trace(X)/trace(dDtmp);
  X = X-diag(diag(dDtmp))*mu1;
end
D0 = Q*diag(fe)*Q';
D1 = Q*X*Q';
