!234567890
C 输出从英制单位改为输出为mm
      PROGRAM CLBL
      !IMPLICIT NONE
      COMMON/CONST/ RE,DX,DY,NCON
      COMMON/CON/ XMIF,TINF,XM1
      COMMON/RST/ UE,DUEDX,PR

      DIMENSION U(10001),U0(10001),V(10001),V0(10001),H(10001),H0(10001)
      DIMENSION A(10001),B(10001),C(10001),R(10001)
      DIMENSION CMU(10001),RHO(10001),RHN(10001),CF(10001),ST(10001)
      DIMENSION GAMMA(10001),TEMP(10001)
      XI=0.0
      GAMMA(1)=1.4
	XF=0.2
      TW=2.4765
      WRITE(*,*) 'PROGRAM CLBL'
      WRITE(*,*) '2-D LAMINAR COMPRESSIBLE AIR BOUNDARY LAYER'
      WRITE(*,*) '1ST ORDER IMPLICIT,LAGGED COEFF'
      WRITE(*,*) 'please input TINF(K),Twall(k),Mainf(Mach),Re(/m)'
      READ*,TINF,TW,XMIF,RE
      WRITE(*,*) 'length of the plate(m)'
      READ*,XF

      
C      TINF=101.98763
C     ! UINF=580.9792
C     ! XMIF=2.87
      NMAX=10001

      MMAX=10001
C!     PINF=0.100365773
C   !   CNUINF=0.00002063252!!0.0000205003933!
      DY=0.000001
      DX=(XF-XI)/(NMAX-1)
	WRITE(*,*) 'DX=',DX
C     ! RE=UINF*1.0/CNUINF
      WRITE(*,*) 'INPUT MMAX,NMAX,RE,DY,DX'
      WRITE(*,*) MMAX,NMAX,RE,DY,DX
      WRITE(*,*) 'XMIF,TINF'
      WRITE(*,*) XMIF,TINF
      PR=0.720
      T1=110.33/TINF
	

      XM1=(GAMMA(1)-1.0)*XMIF**2
      UE=1.0
      DUEDX=0.0
      WRITE(*,*) 'FLAT PLATE FLOWS: UE AND DUEDX'
      WRITE(*,*) UE,DUEDX


      TE=1.0
      HE=TE/XM1
      RHOE=1.0
      PE=1.0/(GAMMA(1)*XMIF**2)
	
      DO  10 M=1,MMAX
      U(M)=UE
      U0(M)=UE
      V(M)=0.0
      V0(M)=0.0
      H(M)=HE
      H0(M)=HE	
      TREF=XM1*H0(M)
      TEMP(M)=TREF
      GAMMA(M)=GAMMA(1)
      CMU(M)=(1.0+T1)*TREF**1.5/(TREF+T1)
      RHO(M)=1.0/(XM1*H(M))
      RHN(M)=RHO(M)
   10 CONTINUE
      U(1)=0.0
      U0(1)=0.0
      H(1)=TW/XM1
      H0(1)=H(1)
      TREF=XM1*H0(1)
      TEMP(1)=TREF
      CMU(1)=(1.0+T1)*TREF**1.5/(TREF+T1)
      RHO(1)=1.0/(XM1*H(1))
      RHN(1)=RHO(1)
	
      MEST=2
      MMM=MEST+3
      DO 20 M=2,MEST
      U0(M)=UE
      V0(M)=0.0
      H0(M)=HE
   20 RHO(M)=1.0/(XM1*H0(M))

      CF(1)=(4.0*U0(2)-U0(3))*CMU(1)/(UE*UE*DY*RE*RHOE)

      RR=PR**0.5

      HAW=HE*(1.0+0.195*RR*XMIF**2)
      ST(1)=(3.0*H0(1)-4.0*H0(2)+H0(3))*CMU(1)
     1/(2.0*PR*UE*DY*RE*RHOE*(H(1)-HAW))
	!PRINT*,CF(1),ST(1),'IM WRITING HERE'
	!PRINT*,V
      DO 200 N=2,NMAX
      NNX=N
      U0(MMAX)=UE
      H0(MMAX)=HE
      RHO(MMAX)=1.0/(XM1*H0(MMAX))

      B(1)=1.0
      C(1)=0.0
      R(1)=0.0

      A(MMAX)=0.0
      B(MMAX)=1.0
      R(MMAX)=UE
      DENO=RE*DY*DY

      DO 50 M=2,MMAX-1
      A(M)=-0.5*RHO(M)*V0(M)/DY-CMU(M-1)/DENO
      B(M)=RHO(M)*U0(M)/DX+0.5*(CMU(M-1)+CMU(M))*2.0
     1/DENO
      C(M)=0.5*RHO(M)*V0(M)/DY-CMU(M)/DENO
      R(M)=RHOE*UE*DUEDX+RHO(M)*U0(M)*U0(M)/DX
   50 CONTINUE
      CALL TRID(MMAX,A,B,C,R,U)

      B(1)=1.0
      C(1)=0.0
      R(1)=H0(1) 

      A(MMAX)=0.0
      B(MMAX)=1.0
      R(MMAX)=HE
      DENO=PR*RE*DY*DY

      DO 60 M=2,MMAX-1
      A(M)=-0.5*RHO(M)*V0(M)/DY-CMU(M-1)/DENO
      B(M)=RHO(M)*U0(M)/DX+0.5*(CMU(M-1)+CMU(M))*2.0
     1/DENO
      C(M)=0.5*RHO(M)*V0(M)/DY-CMU(M)/DENO
      R(M)=-RHOE*UE*DUEDX*U0(M)+RHO(M)*U0(M)*H0(M)/DX
     1+CMU(M)*(U(M+1)-U(M-1))*(U(M+1)-U(M-1))
     2/(4.0*RE*DY*DY)
   60 CONTINUE

      CALL TRID(MMAX,A,B,C,R,H)
      MT=MEST-8
      IF(MT.LT.1) MT=1
      DO 90 M=MT,MMAX
      IF(U(M).GT.0.99*UE) THEN
      MEST=M
      MMM=MEST+3
      GO TO 120
      ENDIF
   90 CONTINUE
  120 CONTINUE

      CALL INTERP(NNX,MMM,GAMMA,TEMP,H)
	!GAMMA(:)=1.4
	!PRINT*,'TEMPERATURE',TEMP
	!PRINT*,GAMMA

      DO 65 M=2,MMAX-1
      CMU(M)=(1.0+T1)*(0.5*(TEMP(M)+TEMP(M-1)))**1.5
     1/(0.5*(TEMP(M)+TEMP(M-1))+T1)
   65 RHN(M)=PE*GAMMA(M)/(0.5*(H(M-1)+H(M))*(GAMMA(M)-1.0))
      DO 80 M=2,MMAX-1
      RHOUA=RHO(M)*U0(M)
      RHOUB=RHO(M-1)*U0(M-1)
      R1=RHN(M-1)/RHN(M)
      V(M)=R1*V(M-1)-(0.5*DY/DX)*(U(M)+R1*U(M-1)-(RHOUA+RHOUB)/RHN(M))
      IF(U(M).EQ.U(M-1).AND.U0(M).EQ.U0(M-1).AND.V(M).EQ.V(M-1))
     1V(M)=0.0
   80 CONTINUE

      DO 110 M=2,MMAX
      U0(M)=U(M)
      V0(M)=V(M)
      H0(M)=H(M)
  110 RHO(M)=RHN(M)

      CF(N)=(4.0*U0(2)-U0(3))*CMU(1)/(UE*UE*DY*RE*RHOE)
      ST(N)=(3.0*H0(1)-4.0*H0(2)+H0(3))*CMU(1)
     1/(2.0*PR*UE*DY*RE*RHOE*(H(1)-HAW))
  200 CONTINUE
      WRITE(*,*) 'FINAL PROFILES OF U AND V'
      CALL OUTPUT(MMAX,RHO,U,V,H)
      WRITE(*,*) 'N CF ST'
C      WRITE(*,320) (N,CF(N),ST(N),N=1,NMAX)
	open(1,file='CF_BL.dat')
	write(1,*) 'variables=x(mm),cf'
	write(1,*) 'zone i=',10001
	do i=1,10001
	write(1,*) DX*REAL(i-1)*1000.0,CF(i)
	!write(1,3) Y(i),U(i),V(i),RHO(i)
    !3 FORMAT(1X,F10.5,1X,F10.5,1X,F10.5,1X,F10.5)
	enddo
	close(1)
  

  320 FORMAT(1X,I3,1X,F10.5,1X,F10.5)
      STOP
      END

      SUBROUTINE INTERP(NNX,MM,GAMMA,TEMP,H)
      COMMON/CON/ XMIF,TINF,XM1
      DIMENSION XT(8),YGAM(8)
      DIMENSION H(10001),GAMMA(10001),TEMP(10001)

      DATA XT/273.0,373.0,473.0,573.0,673.0,773.0,873.0,973.0/
      DATA YGAM/1.401,1.397,1.390,1.378,1.368,1.357,1.346,1.338/
      DO 100 M=2,MM+10
      J=0
      TEMP(M)=XM1*H(M)*TINF
C   80 J=J+1
C      XINT=TEMP(M) 
C      IF(J.GT.160) GO TO 70
C      YOUT=0.0
C      DO 20 K=1,8
C      TERM=YGAM(K)
C      DO 10 L=1,8
C      IF(K.NE.L) TERM=TERM*(XINT-XT(L))/(XT(K)-XT(L))
C   10 CONTINUE
C      YOUT=YOUT+TERM
C   20 CONTINUE
C      TEMP(M)=(1.4/YOUT)*(YOUT-1.0)*XMIF**2*H(M)*TINF
C      IF(ABS(TEMP(M)-XINT).LE.0.005) GO TO 70
C      GO TO 80
C   70 CONTINUE
      GAMMA(M)=1.4!YOUT
      TEMP(M)=TEMP(M)/TINF

  100 CONTINUE
      RETURN
      END

      SUBROUTINE TRID(MM,A,B,C,R,S)
      DIMENSION A(10001),B(10001),C(10001),R(10001)
      DIMENSION S(10001),RP(10001),GAM(10001)

      GAM(1)=C(1)/B(1)
      RP(1)=R(1)/B(1)

      DO 40 M=2,MM
      DENO=B(M)-A(M)*GAM(M-1)
      GAM(M)=C(M)/DENO
      RP(M)=(R(M)-A(M)*RP(M-1))/DENO
   40 CONTINUE
      S(MM)=RP(MM)
      DO 60 M=1,MM-1
      S(MM-M)=RP(MM-M)-GAM(MM-M)*S(MM-M+1)
   60 CONTINUE
      RETURN
      END

      SUBROUTINE OUTPUT(MMAX,RHO,U,V,H)
      COMMON/CONST/ RE,DX,DY,NCON
      COMMON/CON/ XMIF,TINF,XM1
	real(kind=4)::gama1
      DIMENSION RHO(10001),U(10001),V(10001),H(10001),Y(10001)
	gama1=1.40
	XM1=(gama1-1.0)*XMIF**2
	i=1
	!print*,"H(i)*(gama1-1.0)*XMIF**2",H(i)*(gama1-1.0)*XMIF**2
	!print*,"H(1)*0.4*2.86*2.86",H(i)*0.4*2.86*2.86

      DO 15 M=1,MMAX
   15 Y(M)=(M-1)*DY
C	WRITE(*,*) (M,Y(M),U(M),V(M),RHO(M),M=1,MMAX,2)
      WRITE(*,1)
C      WRITE(*,2) (M,Y(M),U(M),V(M),RHO(M),M=1,MMAX)
    1 FORMAT(//,'M',6X,'Y',10X,'U',10X,'V',10X,'RHO',/)
    2 FORMAT(1X,I3,1X,F10.5,1X,F10.5,1X,F10.5,1X,F10.5)
	open(1,file='uprofile.dat')
	write(1,*) 'variables=y(mm),d,u,v,T'
	write(1,*) 'zone i=',10001
	do i=1,10001
	!if(i.eq.1)then
C For Fortran77 this is wrong!	write(1,*), Y(i)*1000.0/25.4,RHO(i),U(i),V(i),H(i)*(gama1-1.0)*XMIF**2
	!endif
	write(1,*) Y(i)*1000.0,RHO(i),U(i),V(i),H(i)*XM1

	!write(1,3) Y(i),U(i),V(i),RHO(i)
    !3 FORMAT(1X,F10.5,1X,F10.5,1X,F10.5,1X,F10.5)
	enddo
	close(1)
      RETURN
      END

	


     

      


	

	




