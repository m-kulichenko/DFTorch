function [PairForces,ERep] = PairPotentialForces(RX,RY,RZ,LBox,Element_Type,Nr_atoms,PairPotType1,PairPotType2,PotCoef)

UNIVPHI = 0;
CUTPHI = 0;

VIRUNIV = 0;
VIRCUT = 0;

for i = 1:Nr_atoms
  FUNIV = zeros(3,1);
  FCUT = zeros(3,1);
  Ra(1) = RX(i); Ra(2) = RY(i); Ra(3) = RZ(i);
  for j = 1:Nr_atoms
  if i ~= j

    for k = 1:10
      if ((Element_Type(i) == PairPotType1(k) && Element_Type(j) == PairPotType2(k)) || (Element_Type(j) == PairPotType1(k) && Element_Type(i) == PairPotType2(k)))
        PPSEL = k;
        R1 = PotCoef(9,PPSEL);
        RCUT = PotCoef(10,PPSEL);
        RCUT2 = RCUT*RCUT;
      end
    end

    RXb = RX(j); RYb = RY(j); RZb = RZ(j);
    for nr_shift_X = -1:1  % Periodic BC shifts in X, Y and Z. Costs a lot extra!
    for nr_shift_Y = -1:1
    for nr_shift_Z = -1:1

      Rb(1) = RXb + nr_shift_X*LBox(1); % Shifts for PBC
      Rb(2) = RYb + nr_shift_Y*LBox(2);
      Rb(3) = RZb + nr_shift_Z*LBox(3);
      Rab = Rb-Ra;  % OBS b - a !!!
      dR = norm(Rab) ;
      dR2 = dR*dR;

      if (dR < RCUT)

         DC = Rab'/dR;

         if (dR < R1)

            POLYNOM = dR*(PotCoef(2,PPSEL) + dR*(PotCoef(3,PPSEL) + dR*(PotCoef(4,PPSEL) + dR*PotCoef(5,PPSEL))));
            PHI = PotCoef(1,PPSEL)*exp(POLYNOM);
            DPOLYNOM = PotCoef(2,PPSEL) + dR*(2*PotCoef(3,PPSEL) + dR*(3*PotCoef(4,PPSEL) + 4*PotCoef(5,PPSEL)*dR));
            DPHI = -DC*PHI*DPOLYNOM;
            EXPTMP = PotCoef(6,PPSEL)*exp( PotCoef(7,PPSEL)*(dR - PotCoef(8,PPSEL)) );

            UNIVPHI = UNIVPHI + PHI + EXPTMP;
            FTMP = DC*PotCoef(7,PPSEL)*EXPTMP;
            FUNIV = FUNIV - DPHI + FTMP;

         else

            MYR = dR - R1;
            CUTPHI =  CUTPHI + PotCoef(11,PPSEL) + MYR*(PotCoef(12,PPSEL) + MYR*(PotCoef(13,PPSEL) + MYR*(PotCoef(14,PPSEL) + MYR*(PotCoef(15,PPSEL) + MYR*PotCoef(16,PPSEL)))));
            FORCE = PotCoef(12,PPSEL)  + MYR*(2*PotCoef(13,PPSEL) + MYR*(3*PotCoef(14,PPSEL) + MYR*(4*PotCoef(15,PPSEL) + MYR*5*PotCoef(16,PPSEL))));
            FCUT = FCUT + DC*FORCE;

         end
      end
    end
    end
    end
  end
  end
  PairForces(:,i) = FUNIV + FCUT;
end
ERep = 0.5*(UNIVPHI + CUTPHI);
