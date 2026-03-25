% EKB PARAMETERS

alpha_1 = 21;  % per_micrometre (in parameters)
beta_1 = 0.94; % millinewton (in parameters)
alpha_2 = 14.6;  % per_micrometre (in parameters) 14.6 17.2
beta_2 = 0.0018;  % millinewton (in parameters) 0.0018 25
alpha_3 = 33.79;  % per_micrometre (in parameters)
beta_3 = 0.0084;  % millinewton (in parameters)

llambda = 55.0;  % millinewton (in parameters)

q_1 = 0.0173;  % per_second (in parameters_izakov_et_al_1991)
q_2 = 0.259;  % per_second (in parameters_izakov_et_al_1991)
q_3 = 0.0173;  % per_second (in parameters_izakov_et_al_1991)
q_4 = 0.015;  % per_second (in parameters_izakov_et_al_1991)

v_max = 0.0055;  %  micrometre_per_second (in parameters)

a = 0.25;  % dimensionless (in parameters)
alpha_Q = 10.0;  % dimensionless (in parameters_izakov_et_al_1991)
beta_Q = 5.0;  % dimensionless (in parameters_izakov_et_al_1991)

x_st = 0.964285;  % dimensionless (in parameters_izakov_et_al_1991)
alpha_G = 1.0;  % dimensionless (in parameters_izakov_et_al_1991)

m_0 = 0.9;  % dimensionless (in parameters)
g_1 = 0.6;  % per_micrometre (in crossbridge_kinetics)
g_2 = 0.52;  % dimensionless (in crossbridge_kinetics)

S_0 = 1.14;  % micrometre (in parameters_izakov_et_al_1991)

chi_1 = 0.55;  % dimensionless (in parameters)
% pi_min = 0.02;

chi_2 = 0.0;

% r0 = 1.7938850613746; % preload
r0 = 0.12; %0.07910141219695446; % preload
% r0 = 2.12 0.091 0.081

d_h = 0.5;  % dimensionless (in parameters)
m = 1.7; %WHATA
% chi_0 = 1.4;  % dimensionless (in parameters)
chi_0 = 2.1;  % dimensionless (in parameters)

alpha_P = 4.0;  % dimensionless (in parameters)
q_st = 1000.0; %WHATA


alpha_vp_l = 16.0;  % per_micrometre (in CE_velocity)
alpha_vp_s = 8;  % per_micrometre (in CE_velocity)
beta_vp_l = 0.00084 * 0.9;  % millinewton_second_per_micrometre (in CE_velocity)
beta_vp_s = 0.84;  % millinewton_second_per_micrometre (in CE_velocity)

a_off = 0.17;  % per_second (in intracellular_calcium_concentration) %changed
a_on = 35.0;

B_1_tot = 0.0;  % millimolar (in intracellular_calcium_concentration)
B_2_tot = 0.0;  % millimolar (in intracellular_calcium_concentration)

% a_eqmin = 0.00128491259225;
% a_eqmin = 0.001299042;
a_eqmin = 0.001299042;
% a_eqmin = 0.00278;
tau_inf = 1500;
% tau_inf = 2.x`

% length = 74.0;  % micrometre (in intracellular_calcium_concentration)

k_mu = 0.6 ; % dimensionless (in parameters)
mu = 3.3;  % dimensionless (in parameters)

s_c = 1.0;

% radius = 12.0;  % micrometre (in intracellular_calcium_concentration) %changed was 12
% n_NaK = 1.5; % dimensionless (in intracellular_sodium_concentration)


% k_A = 40.0;  % per_millimolar (in parameters_izakov_et_al_1991)
k_A = 28.0;  % per_millimolar (in parameters_izakov_et_al_1991) CHANGED

A_tot = 0.07;  % millimolar (in intracellular_calcium_concentration)

%ATP_N = 1; % dimentionless (ingib. factor for N of cross-bridges during ischemia)

n1_A = 0.5;
n1_B = 55.0;
n1_C = 1.0;
n1_Q = 0.835;
n1_K = 1.0;
n1_nu = 5.0;

s055 = 0.55;
s046 = 0.46;

nondimension = 1;
L_0 = 1.67;
R_0 = 1.05;
Restlength = L_0;

if nondimension
    alpha_1 = alpha_1*Restlength;
    alpha_2 = alpha_2*Restlength;
    alpha_3 = alpha_3*Restlength;
    beta_Q = beta_Q*Restlength;
    alpha_vp_l = alpha_vp_l*Restlength;
    beta_vp_l =  beta_vp_l*Restlength;
    beta_vp_s = beta_vp_s*Restlength;
    g_1 = g_1*Restlength;
    n1_B = n1_B*Restlength;
    v_max = v_max/Restlength;
    S_0 = S_0/Restlength;
    s055 = s055/Restlength;
    s046 = s046/Restlength;
end

%-------------------------------------------------------------------------------
% Parameters tkani
%-------------------------------------------------------------------------------

global jj cell_cur Lam_mech N_elec l1 l1_n l2 l3 N_meh n L dx IschemiaDeg BZ1Start BZ1End BZ2Start BZ2End;
% Время
t0=0; % Начальный момент времени (мс)
ts=2000; % Конечный момент времени расчета (мс)
s=10000; % Кол-во расчетных точек
% ts=5000; % Конечный момент времени расчета (мс)
% s=10000; % Кол-во расчетных точек
% s=20000; % Кол-во расчетных точек
%%%%%%%%%%%%%%%%%%
% L Включена!!!%%
%%%%%%%%%%%%%%%%%%
%-------------------------------------------------------------------------------
% Пространство

x0=0;
n=120; % 2) 40; 3) 120
xn=1;
%-------------------------------------------------------------------------------

D=150; % Коэф. диффузии 
L_tkani = 20; % Длина ткани (волокна) мм 2) 20 мм, 3) 60 мм
D_odez=D/(L_tkani)^2;

%-------------------------------------------------------------------------------
% Proc_fib = 100; % Процент фиброза (%)
% num_fib = ceil(n/100*Proc_fib);
% array_fib = ceil(1 + (n-1).*rand(num_fib,1));
% %------------------------------------------------------------------------------
% xx = 1:n;
% sigma = 5;
% mu1 = 40;
% yyy = pdf('Normal',xx,mu1,sigma);
% yy = yyy/max(yyy);
%-------------------------------------------------------------------------------

x=x0:(xn-x0)/(n-1):xn;
t=t0:(ts-t0)/(s-1):ts;
dt=t(2)-t(1);
dx=x(2)-x(1);

%-------------------------------------------------------------------------------

Y=zeros(s,n,24);
Y1=zeros(n,24);
Y0=zeros(s,1,24);
cell_currents = zeros(s,n,14); %All currents in all cells
cell_cur=zeros(14,1);

%-------------------------------------------------------------------------------

l_1=zeros(s,n);
l_2=zeros(s,n);
l_3=zeros(s,1);
N=zeros(s,n);
bzdegree=zeros(n,1); % dimensionless (ingib. factor for N of cross-bridges during ischemia)
v=zeros(s,n);
w=zeros(s,n);
deltaU=zeros(s,n);

%-------------------------------------------------------------------------------
loadfromfileflag = 1;
% loadfromfileflag = 0;
if loadfromfileflag
%     Y(1,:,:)=readmatrix('PhV-120-Norm(NO_MECH-1Hz_ro12_Diff150).xlsx');
%     Y(1,:,:)=readmatrix('PhV-120-Norm(2Hz_ro12_Diff150).xlsx');
%     Y(1,:,:)=readmatrix('PhV-120-Norm(ro12_Diff150).xlsx');
    Y(1,:,:)=readmatrix('PhV-120-20mm-N-10min(30)-N(GZ20_ro12_D150_LCC)_lam-55.xlsx');
%     Y(1,:,:)=readmatrix('PhV-120-20mm-N-10min(10_45-55)-N(GZ10_ro12_D150_LCC)_lam-80.xlsx');
%     Y(1,:,:)=readmatrix('PhV-120-20mm-N-15min(10_45-55)-N(GZ10_ro12_D150_LCC)_lam-80(1).xlsx');
%     Y(1,:,:)=readmatrix('MECH_PhV-120-20mm-N-10min(25)-N(1Hz_GZ10_ro12_D150)_lam-55.xlsx');
%     Y(1,:,:)=readmatrix('NO_MECH_Ko_99PhV-120-20mm-N-15min(25)-N(1Hz_GZ10_ro12_D150_LCCdec)_lam-80.xlsx');
%     Y(1,:,:)=readmatrix('MECH_PhV-120-20mm-N-15min(25)-N(1Hz_GZ10_ro12_D150)_lam-80.xlsx');
%'PhV_120-N_10min20_N(BZ20,20sec).xlsx'
% PhV-120-Norm(ro12).xlsx
% PhV-120-20mm-N-5min(40)-N(GZ10_ro081)_lam-40.xlsx
%     Y(1,:,:)=readmatrix('PhV-120-20mm-N-15min(40)-N(GZ10_ro081)_lam-40_block.xlsx');
%       Y(1,:,:)=readmatrix('PhV-20-Norm(ro081)-10sec.xlsx');
else
    for j=1:n
        Y(1,j,:)= Y_init();
    end
end
%-------------------------------------------------------------------------------
%for jj=1:n
%    Y(1,jj,:)= Y_init();
%end

%Set Special conditions
%Ischemia (0, 5, 10, 15 min) or Border zone values
% IschemiaDeg=0; 
% IschemiaDeg=5; 
% IschemiaDeg=10; 
IschemiaDeg=15; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%See TNNP - Ko(15min) changed!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% IZ length 45 BZ length 10
BZ1Start=25; %54
BZ1End=45; %66
BZ2Start=75; %20; %54
BZ2End=95; %21; %66

% BZ1Start=30; 
% BZ1End=40; 
% BZ2Start=80; 
% BZ2End=90; 
%Here we change ischemia initial values for phase variables
%Model parameters are changed in TNNPE module

for jj=1:n
    %Ischemia zone
    %!!!DON'T FORGET TO CHANGE in TNNPE module!!!!!
%     if jj<n/2
    if jj<(BZ1Start+BZ2End)/2
        bzdegree(jj)=(jj-BZ1Start)/(BZ1End-BZ1Start); %Norm to ischemia
    else
        bzdegree(jj)=(BZ2End-jj)/(BZ2End-BZ2Start); %ischemia to norm
    end
    if bzdegree(jj)>1
        bzdegree(jj)=1;
    end
    if bzdegree(jj)<0
        bzdegree(jj)=0;
    end
%     switch IschemiaDeg
%         case 5
%             ATP_N(jj)=ATP_N(jj)*(1-0.35*bzdegree);
%             %Y(1,jj,25)=Y(1,jj,25)*(1-0.2*bzdegree); %   {30}  {1}	'ATPi',
%             %Y(1,jj,26)=Y(1,jj,26)*(1-0.2*bzdegree); %    {31} {16}	'ADPm',
%             %Y(1,jj,39)=Y(1,jj,39)*(1-0.2*bzdegree); %   {44} {44}	'ATPi_cyto',
%             % Y(1,jj,10)=Y(1,jj,10)*(1-0.05*bzdegree); %   {44} {44}	'K_i',
%         case 10
%             ATP_N(jj)=ATP_N(jj)*(1-0.37*bzdegree);
%             %Y(1,jj,25)=Y(1,jj,25)*(1-0.37*bzdegree); %   {30}  {1}	'ATPi',
%             %Y(1,jj,26)=Y(1,jj,26)*(1-0.37*bzdegree); %    {31} {16}	'ADPm',
%             %Y(1,jj,39)=Y(1,jj,39)*(1-0.37*bzdegree); %   {44} {44}	'ATPi_cyto',
%             % Y(1,jj,10)=Y(1,jj,10)*(1-0.09*bzdegree); %   {44} {44}	'K_i', 
%         case 15
%             ATP_N(jj)=ATP_N(jj)*(1-0.53*bzdegree);
%             %Y(1,jj,25)=Y(1,jj,25)*(1-0.53*bzdegree); %   {30}  {1}	'ATPi',
%             %Y(1,jj,26)=Y(1,jj,26)*(1-0.53*bzdegree); %    {31} {16}	'ADPm',
%             %Y(1,jj,39)=Y(1,jj,39)*(1-0.53*bzdegree); %   {44} {44}	'ATPi_cyto',
%             % Y(1,jj,10)=Y(1,jj,10)*(1-0.137*bzdegree); %   {44} {44}	'K_i',
%     end
end


v(1,:) = -0.00000163008453003026; %2.74398486397373e-06;
w(1,:) = 7.62412076632331e-07;
l_2(1,:) = log((r0+beta_2)/beta_2)/alpha_2;
l_1(1,:) = l_2(1,:) +(log(beta_1)-log(r0+beta_1-beta_2*(exp(alpha_2*l_2(1,:))-1)))/alpha_2;
l_3(1) = log((beta_3+r0)/beta_3)/alpha_3;
N(1,:) = 0.0000284517486098194; %5.6528465136942426e-06;
N_elec = 0;

L = (n-1)*l_2(1,1)*dx+l_3(1);
 
tic
 
v_1 = v_max/10.0;
gamma2 = a*d_h*(0.1)^2.0/(3.0*a*d_h-(a+1.0)*0.1);
v_st = x_st*v_max;

 for i=1:s-1 %main cycle (i - time)
     
%--------------------------------------------------------------------------
    for jj=2:n %1st half-cycle of electric (jj - # of cell)
        N_elec=N(i,jj);
        [T1,X1] = ode15s(@TNNPE,[t(i) t(i)+dt/2],Y(i,jj,:));
        Y1(jj,:)=X1(end,:);
    end
    N_elec=N(i,1);
    Y0=Y(i,1,:);
    [Yqn,V_1]=pde(t(i),t(i+1),x0,xn,n,D_odez,Y1,Y0);
    cell_currents(i,1,:) = cell_cur(:);
    Y1(:,13)=Yqn;
    %рассчитываем разность ПД между соседними клетками (jj) в момент
    %времени i
    for jj=2:n
        deltaU(i,jj)=Yqn(jj)-Yqn(jj-1);
    end
    for jj=2:n %2nd half-cycle of electric (jj - # of cell)
        N_elec=N(i,jj);
        [T2,X2] = ode15s(@TNNPE,[t(i)+dt/2 t(i+1)],Y1(jj,:));
        Y(i+1,jj,:)=X2(end,:);
    end
    for jj=1:n
        cell_currents(i,jj,:) = cell_cur(:);
    end
    Y(i+1,1,:)=V_1;

%--------------------------------------------------------------------------
    for j=1:n

        switch IschemiaDeg
            case 5
                Lam_mech = llambda*(1-0.43636*bzdegree(j));
            case 10
                Lam_mech = llambda*(1-0.55*bzdegree(j));
            case 15
                Lam_mech = llambda*(1-0.8*bzdegree(j));
        end


        if (v(i,j) <= 0.0)
            q_v = q_1-q_2*v(i,j)/v_max;
        elseif (v(i,j) <= v_st)
            q_v = (q_4-q_3)*v(i,j)/v_st+q_3;
        else
            q_v = q_4/(1.0+beta_Q*(v(i,j)-v_st)/v_max)^alpha_Q;
        end
        
        if (v(i,j) <= 0.0)
           P_star = a*(1.0+v(i,j)/v_max)/(a-v(i,j)/v_max);
        else
           P_star = 1.0+d_h-d_h^2.0*a/(a*d_h/gamma2*(v(i,j)/v_max)^2.0+(a+1.0)*v(i,j)/v_max+a*d_h);
        end

        if (v(i,j) <= 0.0)
           G_star = 1.0+0.6*v(i,j)/v_max;
        elseif (v(i,j) <= 0.1)
           G_star = P_star/((0.4*a+1.0)*v(i,j)/(a*v_max)+1.0);
        else
           G_star = P_star*exp(-alpha_G*((v(i,j)-v_1)/v_max)^alpha_P)/((0.4*a+1.0)*v(i,j)/(a*v_max)+1.0);
        end
        
        if (v(i,j) <= 0.0)
            chi = chi_1 + chi_2 * v(i,j) / v_max;
        else
            chi = chi_1;
        end
        
        k_p_v = chi*chi_0*q_v*m_0*G_star;
        M_A = (Y(i+1,j,12)/A_tot)^mu*(1.0+k_mu^mu)/((Y(i+1,j,12)/A_tot)^mu+k_mu^mu);

        if ((g_1*l_1(i,j)+g_2)*(n1_A+(n1_K-n1_A)/(n1_C+n1_Q*exp(-n1_B*l_1(i,j)))^(1 / n1_nu))< 0.0)
           n_1 = 0.0;
        elseif ((g_1*l_1(i,j)+g_2)*(n1_A+(n1_K-n1_A)/(n1_C +n1_Q*exp(-n1_B*l_1(i,j)))^(1/n1_nu))<1.0)
           n_1 = (g_1*l_1(i,j)+g_2)*(n1_A+(n1_K-n1_A)/(n1_C+n1_Q*exp(-n1_B*l_1(i,j)))^(1/n1_nu));
        else
           n_1 = 1.0;
        end

        if (l_1(i,j) <= s055)
           L_oz = (l_1(i,j)+S_0)/(s046+S_0);
        else
           L_oz = (S_0+s055)/(s046+S_0);
        end
        
        if (v(i,j) <= -v_max)
            p_ext = 0.0;
        elseif (v(i,j) <= 0.0)
            p_ext = a*(1.0+v(i,j)/v_max)/((a-v(i,j)/v_max)*(1.0+0.6*v(i,j)/v_max));
        elseif (v(i,j) <= v_1)
            p_ext = (0.4*a+1.0)*v(i,j)/(a*v_max)+1;
        else
            p_ext = ((0.4*a+1.0)*v(i,j)/(a*v_max)+1.0)*exp(alpha_G*(v(i,j)/v_max-0.1)^alpha_P);
        end

        p_v = p_ext;

        k_m_v = chi_0*q_v*(1.0-chi*m_0*G_star);
        K_chi = k_p_v*M_A*n_1*L_oz*(1.0-N(i,j))-k_m_v*N(i,j);
        %p_v = P_star/G_star;
        
        N(i+1,j)=N(i,j)+dt*K_chi;
        
        l1=l_1(i,j);
        l2=l_2(i,j);
        l3=l_3(i);
        N_meh=N(i+1,j);
        options = optimset('Display','off');
        [v_j,~] = fsolve(@v_meh,v(i,j),options);
        v(i+1,j)=v_j;
        %%% L calculation is ON
        l_1(i+1,j)=l_1(i,j)+dt*v(i+1,j);
        %%% L calculation is OFF
%         l_1(i+1,j)=l_1(i,j);%+dt*v(i+1,j);
        l1_n(j)=l_1(i+1,j);
    end
    l2n_l3=[l_2(i,:) l_3(i)];
    [l2_l3,~] = fsolve(@l2l3,l2n_l3,options);
    for j=1:n
        l_2(i+1,j)=l2_l3(j);
    end
    l_3(i+1)=l2_l3(n+1);
 end
 toc