function dY = TNNPE(time, Y)

%-------------------------------------------------------------------------------
% Constants
%-------------------------------------------------------------------------------
global jj cell_cur N_elec IschemiaDeg n BZ1Start BZ1End BZ2Start BZ2End;

% [[constants]]
% Parameters for the Ca handling
GeneralTnC         = 0.07;          %'TnС общий',
BindingConstTnC    = 1000;          %'Константа связывания TnС',
C20                = 8;             %'Константа распада TnС',
qa                 = 35;            %'qa'

% Physiological mode constants
% k_phys_rel = 0.05;  % [1/ms] physiological relaxation velocity
% per_phys_rel = 230;  % [ms] period of physiological relaxation

% Parameters for the mechanical part
% alpha1             = 21;                 %'alfa1',
% beta1              = 0.94; %0.47         %'beta1', PV
% alpha2             = 14.6;               %'alfa2',
% beta2              = 0.0018;             %'beta2',
% Ekb_lambda         = 55;                 %'lambda',
% q1                 = 17.3;               %'q1',
% q2                 = 259;                %'q2',
% q3                 = 17.3;               %'q3',
% q4                 = 15;                 %'q4',
% vmax               = 5.50e-03;           %'vmax', {скорость укорочения при нулевой нагрузке}
% a                  = 0.25;               %'a',
% alphaQ             = 10;                 %'alfaq',
% betaQ              = 5;                  %'betaq',
% xst                = 0.964285;           %'Xst', {Хst=Vst/Vmax , Vst-скорость при которой резко уменьшается qn}
% alphaG             = 1;                  %'alfag',
% m0                 = 0.9;                %'m0',{начальная величина вероятности прикрепления n2}
% g1                 = 0.6;                %'g1',
% g2                 = 0.52;               %'g2',
% n1A                = 0.5;                %'n1A', //Новые параметры для вычисления n1(l1)
% n1B                = 55;                 %'n1B', //----
% n1C                = 1;                  %'n1C', //----
% n1Q                = 0.835;              %'n1Q', //----
% n1K                = 1;                  %'n1K', //----
% n1Nu               = 0.1;                %'n1Nu',//----
% s0                 = 1.14;               %'s0',
% kappa              = 0.95; %0.9          %'каппа', {изменение циклирования мостика} PV
pimin              = 0.03;               %'pimin',
% r0                 = 0.07910141219695446;%0.0646910072147687 %PK  %'ro',
% Ekb_d              = 0.5;                %'d',
% x1                 = 0.1;                %'x1',
% x0                 = 0.9;                %'x0',
% m                  = 1.7;                %'m' ,
% kappa0             = 3.7e-03;  %3.00e-03 %'каппа0', PV
% alphaP             = 4;                  %'alphap' ,
% qst                = 1000;               % 'qst',
% alpha3             = 33.79;              %'alpha3',
% beta3              = 0.0084; %0.0042     %'beta3', PK
% alpha_tir          = 30;                 %'alpha_tir',
% beta_tir           = 0;                  %'beta_tir',
% l_tir              = -0.03;              %'l_tir',
% vs1                = 0.00084 * 0.9;      %'vs1', {beta_vp участвует в вычислении коэффициента вязкости vs1, масштаб.множитель}
% alp_vp             = 16;                 %'alp_vp', {участвует в вычислении коэффициента вязкости vs1, множитель в эксп.}
% vs1rel             = 8.4e-06;            %'vs1rel',
% alp_vpr            = 8;                  %'alp_vpr',
% vs2                = 0;                  %'vs2', {beta_vs участвует в вычислении коэффициента вязкости vs2, масштаб.множитель}
% alp_vs             = 46;                 %'alp_vs', {участвует в вычислении коэффициента вязкости vs2, множитель в эксп.}
% vs2rel             = 0;                  %'vs2rel',
% alp_vsr            = 39;                 %'alp_vsr',
sc                 = 0.7;                %'sc',
% k_mu               = 0.6;                %'k_mu',
% mu1                = 3;                  %'mu1',
% mu_shift           = 2.55e-03;           %'mu_shift'

% parameters electrical part
K_L              = 0.00022;   %'K_L,mM (in CaRU_Transitions)',
K_RyR            = 0.041;     %'K_RyR,mM (in CaRU_Transitions)',
V_L              = -2;        %'V_L,mV (in CaRU_Transitions)',
a1               = 0.0625;    %'a1,dimensionless (a in CaRU_Transitions)',
b                = 14;        %'b,dimensionless (in CaRU_Transitions)',
c                = 0.01;      %'c,dimensionless (in CaRU_Transitions)',
TNP_d            = 100;       %'d,dimensionless (in CaRU_Transitions)',
del_VL           = 7;         %'del_VL,mV (in CaRU_Transitions)',
phi_L            = 2.35;      %'phi_L,dimensionless (in CaRU_Transitions)',
phi_R            = 0.05;      %'phi_R,dimensionless (in CaRU_Transitions)',
t_L              = 1;         %'t_L,ms (in CaRU_Transitions)',
tau_L            = 650;       %'tau_L,ms (in CaRU_Transitions)',
tau_R_1          = 2.43;      %'tau_R_1,ms (tau_R in CaRU_Transitions)',
theta_R          = 0.012;     %'theta_R,dimensionless (in CaRU_Transitions)',
V_SR_uL          = 2.098e-6;  %PK %'V_SR_uL,uL (in cell_geometry)',
V_myo            = 25850;     %'V_myo,um3 (in cell_geometry)',
V_myo_uL         = 2.585e-5; %PK %'V_myo_uL,uL (in cell_geometry)',
g_CaB            = 2.6875e-8; %PK  %'g_CaB,mM_per_mV_ms (in hinch_Background_Ca_current)',
J_L              = 0.000913;  %'J_L,um3_per_ms (in hinch_CaRU)',
J_R              = 0.02;      %'J_R,um3_per_ms (in hinch_CaRU)',
N_L_CaRU         = 50000;     %'N,dimensionless (in hinch_CaRU)',
N_R_CaRU         = 50000;     % N for RyR channels
g_D              = 0.065;     %'g_D,um3_per_ms (in hinch_CaRU)',
K_mCa            = 1.38;      %'K_mCa,mM (in hinch_Na_Ca_Exchanger)',
K_mNa            = 87.5;      %'K_mNa,mM (in hinch_Na_Ca_Exchanger)',
eta              = 0.35;      %'eta,dimensionless (in hinch_Na_Ca_Exchanger)',
g_NCX            = 0.0385;    %'g_NCX,mM_per_ms (in hinch_Na_Ca_Exchanger)',
k_sat            = 0.1;       %'k_sat,dimensionless (in hinch_Na_Ca_Exchanger)',
K_SERCA          = 0.0005;    %'K_SERCA,mM (in hinch_SERCA)',
g_SERCA          = 0.00045;   %'g_SERCA,mM_per_ms (in hinch_SERCA)',
g_SRl            = 1.8951e-5; %PK  %'g_SRl,per_ms (in hinch_SR_Ca_leak_current)' %1.90e-05
K_mpCa           = 0.0005;    %'K_mpCa,mM (in hinch_Sarcolemmal_Ca_pump)',
g_pCa            = 3.50e-06;  %'g_pCa,mM_per_ms (in hinch_Sarcolemmal_Ca_pump)',
B_CMDN           = 0.05;      %'B_CMDN,mM (in hinch_calmodulin_Ca_buffer)',
k_CMDN           = 0.002382;  %'k_CMDN,mM (in hinch_calmodulin_Ca_buffer)',
Cm               = 0.0001;    %'Cm,uF (in membrane)',
F                = 96487;     %'F,C_per_mole (in membrane)',
R2               = 8314.5;    %'R2,mJ_per_mole_K (R in membrane)',
Tem              = 295;       %'T,kelvin (in membrane)',
stim_amplitude   = -0.0006;   %'stim_amplitude,uA (in membrane)',
stim_duration    = 10 ;       %'stim_duration,ms (in membrane)',
% stim_period      = 500;      %'stim_period,ms (in membrane)',
stim_period      = 1000;      %'stim_period,ms (in membrane)',
StimStart_shift  = 60.0;
%A_1              = 50000;     %'A_1,dimensionless (in niederer_Cross_Bridges)',
%A_2              = 138;       %'A_2,dimensiofnless (in niederer_Cross_Bridges)',
%A_3              = 129;       %'A_3,dimensionless (in niederer_Cross_Bridges)',
%a2               = 0.35;      %'a2,dimensionless (a in niederer_Cross_Bridges)',
%alpha_1          = 0.03;      %'alpha_1,per_ms (in niederer_Cross_Bridges)',
%alpha_2          = 0.13;      %'alpha_2,per_ms (in niederer_Cross_Bridges)',
%alpha_3          = 0.625;     %'alpha_3,per_ms (in niederer_Cross_Bridges)',
%beta_0           = 4.9;       %'beta_0,dimensionless (in niederer_filament_overlap)',
%T_ref            = 56.2;      %'T_ref,N_per_mm2 (in niederer_length_independent_tension)',
%Ca_50ref         = 0.00105;   %'Ca_50ref,mM (in niederer_tropomyosin)',
%K_z              = 0.15;      %'K_z,dimensionless (in niederer_tropomyosin)',
%alpha_0          = 0.008;     %'alpha_0,per_ms (in niederer_tropomyosin)',
%alpha_r1         = 0.002;     %'alpha_r1,per_ms (in niederer_tropomyosin)',
%alpha_r2         = 0.00175;   %'alpha_r2,per_ms (in niederer_tropomyosin)',
%beta_1           = -4;        %'beta_1,dimensionless (in niederer_tropomyosin)',
%n_Hill           = 3;         %'n_Hill,dimensionless (in niederer_tropomyosin)',
%n_Rel            = 3;         %'n_Rel,dimensionless (in niederer_tropomyosin)',
%z_p              = 0.85;      %'z_p,dimensionless (in niederer_tropomyosin)',
%Ca_TRPN_Max      = 0.07;      %'Ca_TRPN_Max,mM (in niederer_troponin)',
%gamma_trpn       = 2;         %'gamma_trpn,dimensionless (in niederer_troponin)',
%k_Ref_off        = 0.2;       %'k_Ref_off,per_ms (in niederer_troponin)',
%k_on             = 100;       %'k_on,per_mM_ms (in niederer_troponin)',
a_to             = 0.883;     %'a_endo,dimensionless (in GattoniNoCAMK)',
b_to             = 0.117;     %'b_endo,dimensionless (in GattoniNoCAMK)',
g_t              = 1.96E-05;  %'g_t,mSi (in GattoniNoCAMK)',
g_B_K            = 1.38e-07;  %'g_B_K,mSi (in pandit_background_currents)',
g_B_Na           = 8.015e-8;  %PK  %'g_B_Na,mSi (in pandit_background_currents)',
f_Na             = 0.2;       %'f_Na,dimensionless (in pandit_hyperpolarisation_activated_current)',
g_f              = 1.45e-06;  %'g_f,mSi (in pandit_hyperpolarisation_activated_current)',
g_K1             = 2.40e-05;  %'g_K1,mSi (in pandit_inward_rectifier)',
g_Na             = 0.0008;    %'g_Na,mSi (in pandit_sodium_current)',
K_m_K            = 1.5;       %'K_m_K,mM (in pandit_sodium_potassium_pump)',
K_m_Na           = 10;        %'K_m_Na,mM (in pandit_sodium_potassium_pump)',
i_NaK_max        = 9.50e-05;  %'i_NaK_max,uA (in pandit_sodium_potassium_pump)',
Ca_o             = 1.2;       %'Ca_o,mM (in pandit_standard_ionic_concentrations)',
K_o              = 5.4;       %'K_o,mM (in pandit_standard_ionic_concentrations)',
Na_o             = 140;       %'Na_o,mM (in pandit_standard_ionic_concentrations)',
g_ss             = 7.00e-06;  %'g_ss,mSi (in pandit_steady_state_outward_K_current)'

g_K_ATP          = 1.150e-3;  % mSi (in NICHOLS&LEDERER 1989)
%g_K_ATP          = 0.0;      % mSi (in NICHOLS&LEDERER 1989)    отключен в этих экспериментах
ATP_i            = 6.8;       % mM  (in Kursanov protocol)
V50_P_ATP        = 0.1;       % mM  (in NICHOLS&LEDERER 1989)
hill_P_ATP       = 2.0;       %     (in NICHOLS&LEDERER 1989)
K_o_norm         = 5.4;       % mM  (in pandit_standard_ionic_concentrations)

deg_Ko_K1        = 0.5;       % [1]

%Here we change ischemia parameters
    %Normal and ischemia zones
% if jj<n/2
if jj<(BZ1Start+BZ2End)/2
    bzdegree=(jj-BZ1Start)/(BZ1End-BZ1Start); %Norm to ischemia
else
    bzdegree=(BZ2End-jj)/(BZ2End-BZ2Start); %ischemia to norm
end
if bzdegree>1
    bzdegree=1;
end
if bzdegree<0
    bzdegree=0;
end
switch IschemiaDeg
    case 5
        %ATP_max=ATP_max*(1-0.2*bzdegree);
        %Cm_ATP=Cm_ATP*(1-0.2*bzdegree);
%         g_t=g_t*(1-0.0*bzdegree);
        ATP_i = ATP_i*(1-0.2*bzdegree); %5.44
%         K_m_Na=K_m_Na*(1+0.5*bzdegree);
%         g_SERCA=g_SERCA*(1-0.15*bzdegree);
        K_o=5.4+1.885*bzdegree;%9.17;
        g_Na=g_Na*(1-0.125*bzdegree);
        J_L=J_L*(1-0.15*bzdegree);
    case 10
        %ATP_max=ATP_max*(1-0.37*bzdegree);
        %Cm_ATP=Cm_ATP*(1-0.37*bzdegree);
%         g_t=g_t*(1-0.5*bzdegree);
        ATP_i = ATP_i*(1-0.37*bzdegree);
%         K_m_Na=K_m_Na*(1+bzdegree);
%         g_SERCA=g_SERCA*(1-0.25*bzdegree);
        K_o=5.4+1.885*bzdegree;%9.17; 3.77
        g_Na=g_Na*(1-0.25*bzdegree);
        J_L=J_L*(1-0.3*bzdegree);
    case 15
        %ATP_max=ATP_max*(1-0.53*bzdegree);
        %Cm_ATP=Cm_ATP*(1-0.53*bzdegree);
%         g_t=g_t*(1-bzdegree);
        ATP_i = ATP_i*(1-0.53*bzdegree);
%         K_m_Na=K_m_Na*(1+1.5*bzdegree);
%         g_SERCA=g_SERCA*(1-0.3*bzdegree);
        K_o=5.4+4.5*bzdegree;%13.37;8.37 4.185 5.7384
        g_Na=g_Na*(1-0.375*bzdegree);
        J_L=J_L*(1-0.7*bzdegree); %0.55
end


exp_00001 = exp(-0.00001);
t_R = 1.17*t_L;
alpha_m = phi_L / t_L;
beta_m = phi_R / t_R;
g_Na_endo = 1.33 * g_Na;
sigma = (exp(Na_o/67.3) - 1.0) / 7.0;
%g_t_endo = 0.4647 * g_t;
f_K = 1.0 - f_Na;
%K_2 = alpha_r2 * (z_p^n_Rel) / ((z_p^n_Rel) + (K_z^n_Rel)) * (1.0 - n_Rel*(K_z^n_Rel)/((z_p^n_Rel) + (K_z^n_Rel)));
%K_1 = alpha_r2 * (z_p^(n_Rel-1.0)) * n_Rel * (K_z^n_Rel) / (((z_p^n_Rel) + (K_z^n_Rel))^2);
%dExtensionRatiodt = 0.0;  % нигде не используется
tau_s_ss = 2100.0;
sqr_ryr = K_RyR^2;

%-------------------------------------------------------------------------------
% Computation
%-------------------------------------------------------------------------------

% time (second)

z_1 = Y(1);  % 'z_1 (dimensionless) (in CaRU_reduced_states)',
z_2 = Y(2);  % 'z_2 (dimensionless) (in CaRU_reduced_states)',
z_3 = Y(3);  % 'z_3 (dimensionless) (in CaRU_reduced_states)',
r1 = Y(4);  % 'r1 (dimensionless) (r in Ca_independent_transient_outward_K_current_r_gate)',
s = Y(5);  % 's (dimensionless) (in Ca_independent_transient_outward_K_current_s_gate)',
s_slow = Y(6);  % 's_slow (dimensionless) (in Ca_independent_transient_outward_K_current_s_slow_gate)',
y = Y(7);  % 'y (dimensionless) (in hyperpolarisation_activated_current_y_gate)',
Ca_SR = Y(8);  % 'Ca_SR (mM) (in intracellular_ion_concentrations)',
Ca_i = Y(9);  % 'Ca_i (mM) (in intracellular_ion_concentrations)',
K_i = Y(10);  % 'K_i (mM) (in intracellular_ion_concentrations)',
Na_i = Y(11);  % 'Na_i (mM) (in intracellular_ion_concentrations)',
TRPN = Y(12);  % 'TRPN (mM) (in intracellular_ion_concentrations)',
V = Y(13);  % 'V (mV) (in membrane)',
%Q_1 = Y(14);  % 'Q_1 (dimensionless) (in niederer_Cross_Bridges)',
%Q_2 = Y(15);  % 'Q_2 (dimensionless) (in niederer_Cross_Bridges)',
%Q_3 = Y(16);  % 'Q_3 (dimensionless) (in niederer_Cross_Bridges)',
%z = Y(17);  % 'z (dimensionless) (in niederer_tropomyosin)',
h = Y(18);  % 'h (dimensionless) (in sodium_current_h_gate)',
j_sod = Y(19);  % 'j (dimensionless) (in sodium_current_j_gate)',
m_iNa = Y(20);  % 'm (dimensionless) (in sodium_current_m_gate)',
r_ss = Y(21);  % 'r_ss (dimensionless) (in steady_state_outward_K_current_r_ss_gate)',
s_ss = Y(22);  % 's_ss (dimensionless) (in steady_state_outward_K_current_s_ss_gate)',
%B1 = Y(23);  % 'B1',
%B2 = Y(24);  % 'B2',


pi_n1 = GeneralTnC * sc * N_elec / (TRPN);
      
if (pi_n1 <= 0.0)
   piv = 1.0;
elseif (pi_n1 <= 1.0)
   piv = pimin^pi_n1;
else
   piv = pimin;
end

koff = C20 * exp(-qa * TRPN) * piv;
dY(12, 1) = BindingConstTnC * (GeneralTnC - TRPN) * Ca_i - koff * TRPN;

% Rat

% Niederer
expVL = exp((V - V_L) / del_VL);
alpha_p = expVL / (t_L * (expVL + 1.0));
FVRT = F * V / (R2 * Tem);
FVRT_Ca = 2.0 * FVRT;
exp_FVRT_Ca = exp(-FVRT_Ca);

if (abs(FVRT_Ca) > 1.0e-9)
    C_oc = (Ca_i + J_L / g_D * Ca_o * FVRT_Ca * exp_FVRT_Ca / (1.0 - exp_FVRT_Ca)) / (1.0 + J_L / g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca));
else
    C_oc = (Ca_i + J_L / g_D * Ca_o) / (1.0 + J_L / g_D);
end

sqr_coc = C_oc * C_oc;
sqr_y8 = Ca_i * Ca_i;
beta_poc = sqr_coc / (t_R * (sqr_coc + sqr_ryr));
beta_pcc = sqr_y8 / (t_R * (sqr_y8 + sqr_ryr));
C_co = (Ca_i + J_R / g_D * Ca_SR) / (1.0 + J_R / g_D);
epsilon_tmp = (expVL + a1) / (tau_L * K_L * (expVL + 1.0));
epsilon_pco = C_co * epsilon_tmp;
epsilon_pcc = Ca_i * epsilon_tmp;
epsilon_m = b * (expVL + a1) / (tau_L * (b * expVL + a1));
mu_poc = (sqr_coc + c * sqr_ryr) / (tau_R_1 * (sqr_coc + sqr_ryr));
mu_pcc = (sqr_y8 + c * sqr_ryr) / (tau_R_1 * (sqr_y8 + sqr_ryr));
mu_moc = theta_R * TNP_d * (sqr_coc + c * sqr_ryr) / (tau_R_1 * (TNP_d * sqr_coc + c * sqr_ryr));
mu_mcc = theta_R * TNP_d * (sqr_y8 + c * sqr_ryr) / (tau_R_1 * (TNP_d * sqr_y8 + c * sqr_ryr));
denom = 1 / ((alpha_p + alpha_m) * ((alpha_m + beta_m + beta_poc) * (beta_m + beta_pcc) + alpha_p * (beta_m + beta_poc)));
y_oc = alpha_p * beta_m * (alpha_p + alpha_m + beta_m + beta_pcc) * denom;
y_cc = alpha_m * beta_m * (alpha_m + alpha_p + beta_m + beta_poc) * denom;
r_1 = y_oc * mu_poc + y_cc * mu_pcc;
r_2 = (alpha_p * mu_moc + alpha_m * mu_mcc) / (alpha_p + alpha_m);
r_3 = beta_m * mu_pcc / (beta_m + beta_pcc);
r_4 = mu_mcc;
y_co = alpha_m * (beta_pcc * (alpha_m + beta_m + beta_poc) + beta_poc * alpha_p) * denom;
r_5 = y_co * epsilon_pco + y_cc * epsilon_pcc;
r_6 = epsilon_m;
r_7 = alpha_m * epsilon_pcc / (alpha_p + alpha_m);
r_8 = epsilon_m;
z_4 = 1.0 - z_1 - z_2 - z_3;
dY(1, 1) = -(r_1 + r_5) * z_1 + r_2 * z_2 + r_6 * z_3;  % Z1
dY(2, 1) = r_1 * z_1 - (r_2 + r_7) * z_2 + r_8 * z_4;  % Z2
dY(3, 1) = r_5 * z_1 - (r_6 + r_3) * z_3 + r_4 * z_4;  % Z3
y_oo = alpha_p * (beta_poc * (alpha_p + beta_m + beta_pcc) + beta_pcc * alpha_m) * denom;
y_ci = alpha_m / (alpha_p + alpha_m);
y_oi = alpha_p / (alpha_p + alpha_m);
y_ic = beta_m / (beta_pcc + beta_m);
y_io = beta_pcc / (beta_pcc + beta_m);
y_ii = 1.0 - y_oc - y_co - y_oo - y_cc - y_ci - y_ic - y_oi - y_io;
r_infinity = 1.0 / (1.0 + exp((V + 10.6) / -11.42));
tau_r_2 = 100.0 / (45.16 * exp(0.03577 * (V + 50.0)) + 98.9 * exp(-0.1 * (V + 38.0))); % in GattoniNoCAMK
dY(4, 1) = (r_infinity - r1) / tau_r_2;  % R1
s_infinity = 1.0 / (1.0 + exp((V + 45.3) / 6.8841));
tau_s = 20.0 * exp(-((V + 70.0) / 25.0) ^ 2) + 35.0; % in GattoniNoCAMK
dY(5, 1) = (s_infinity - s) / tau_s;  % S
s_slow_infinity = 1.0 / (1.0 + exp((V + 45.3) / 6.8841));
tau_s_slow = 1300.0 * exp(-((V + 70.0) / 30.0) ^ 2) + 35.0; % in GattoniNoCAMK
dY(6, 1) = (s_slow_infinity - s_slow) / tau_s_slow;  % S_Slow

C_cc = Ca_i;

if (abs(FVRT_Ca) > 1e-5)
    C_oo = (Ca_i + J_R / g_D * Ca_SR + J_L / g_D * Ca_o * FVRT_Ca * exp_FVRT_Ca / (1.0 - exp_FVRT_Ca)) / (1.0 + J_R / g_D + J_L / g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca));
else
    C_oo = (Ca_i + J_R / g_D * Ca_SR + J_L / g_D * Ca_o) / (1.0 + J_R / g_D + J_L / g_D);
end

J_Rco = J_R * (Ca_SR - Ca_i) / (1.0 + J_R / g_D);

if (abs(FVRT_Ca) > 1e-5)
    J_Roo = J_R * (Ca_SR - Ca_i + J_L / g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca) * (Ca_SR - Ca_o * exp_FVRT_Ca)) / (1.0 + J_R / g_D + J_L / g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca));
else
    J_Roo = J_R * (Ca_SR - Ca_i + J_L / g_D * 0.00001 / (1.0 - exp_00001) * (Ca_SR - Ca_o * exp_00001)) / (1.0 + J_R / g_D + J_L / g_D * 0.00001 / (1.0 - exp_00001));
end
if (abs(FVRT_Ca) > 1e-5)
   J_Loc = J_L * FVRT_Ca / (1.0 - exp_FVRT_Ca) * (Ca_o * exp_FVRT_Ca - Ca_i) / (1.0 + J_L / g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)); 
else
   J_Loc = J_L * 0.00001 / (1.0 - exp_00001) * (Ca_o * exp_00001 - Ca_i) / (1.0 + J_L / g_D * 0.00001 / (1.0 - exp_00001));
end
                
if (abs(FVRT_Ca) > 1e-5) 
    J_Loo = J_L * FVRT_Ca / (1.0 - exp_FVRT_Ca) * (Ca_o * exp_FVRT_Ca - Ca_i + J_R / g_D * (Ca_o * exp_FVRT_Ca - Ca_SR)) / (1.0 + J_R / g_D + J_L / g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca));
else
    J_Loo = J_L * 0.00001 / (1.0 - exp_00001) * (Ca_o * exp_00001 - Ca_i + J_R / g_D * (Ca_o * exp_00001 - Ca_SR)) / (1.0 + J_R / g_D + J_L / g_D * 0.00001 / (1.0 - exp_00001));
end

J_L1 = J_Loo * y_oo + J_Loc * y_oc;
J_L2 = J_Loc * alpha_p / (alpha_p + alpha_m);
I_LCC_1 = (z_1 * J_L1 + z_2 * J_L2) * N_L_CaRU / V_myo; % разделили N_CaRU ! (можно заменить на просто масштаб коэф.)
J_R1 = y_oo * J_Roo + J_Rco * y_co;
J_R3 = J_Rco * beta_pcc / (beta_m + beta_pcc);
I_RyR_1 = (z_1 * J_R1 + z_3 * J_R3) * N_R_CaRU / V_myo; % разделили N_CaRU ! (можно заменить на просто масштаб коэф.)
I_LCC_2 = -1.5 * I_LCC_1 * 2.0 * V_myo_uL * F;
I_NaCa_2 = g_NCX * (exp(eta * FVRT) * (Na_i * Na_i * Na_i) * Ca_o - exp((eta - 1.0) * FVRT) * (Na_o * Na_o * Na_o) * Ca_i) / (((Na_o * Na_o * Na_o) + (K_mNa * K_mNa * K_mNa)) * (Ca_o + K_mCa) * (1.0 + k_sat * exp((eta - 1.0) * FVRT)));
I_NaCa_1 = I_NaCa_2 * V_myo_uL * F;
I_pCa_2 = g_pCa * Ca_i / (K_mpCa + Ca_i);
I_pCa_1 = I_pCa_2 * 2.0 * V_myo_uL * F;
E_Ca = R2 * Tem / (2.0 * F) * log(Ca_o / Ca_i);
I_CaB_2 = g_CaB * (E_Ca - V);
I_CaB_1 = -I_CaB_2 * 2.0 * V_myo_uL * F;
I_RyR_2 = 1.5 * I_RyR_1;
I_SERCA = g_SERCA * sqr_y8 / (K_SERCA * K_SERCA + sqr_y8);
I_SR = g_SRl * (Ca_SR - Ca_i);
beta_CMDN = 1.0 / (1.0 + k_CMDN * B_CMDN / ((k_CMDN + Ca_i) * (k_CMDN + Ca_i)));
y_infinity = 1.0 / (1.0 + exp((V + 138.6) / 10.48));
tau_y = 1000.0 / (0.11885 * exp((V + 80.0) / 28.37) + 0.5623 * exp((V + 80.0) / -14.19));
dY(7, 1) = (y_infinity - y) / tau_y ; % y
E_Na = R2 * Tem / F * log(Na_o / Na_i);
i_Na = g_Na_endo * (m_iNa * m_iNa * m_iNa) * h * j_sod * (V - E_Na);
i_B_Na = g_B_Na * (V - E_Na);
i_NaK = i_NaK_max * 1.0 / (1.0 + 0.1245 * exp(-0.1 * V * F / (R2 * Tem)) + 0.0365 * sigma * exp(-V * F / (R2 * Tem))) * K_o / (K_o + K_m_K) * 1.0 / (1.0 + ((K_m_Na / Na_i) ^ 4.0));
i_f_Na = g_f * y * f_Na * (V - E_Na);
dY(11, 1) = -(i_Na + i_B_Na + I_NaCa_1 * 3.0 + i_NaK * 3.0 + i_f_Na) * 1.0 / (V_myo_uL * F);  % Na_i

% if (((t - floor(t / self.period) * self.period) >= self.start)and ((t - floor(t / self.period) * self.period) <= self.start + stim_duration)):
%     I_Stim = stim_amplitude;
% else
%     I_Stim = 0.0;
% end

if ((time - floor(time/stim_period)*stim_period >= StimStart_shift) &&  (time-floor(time/stim_period)*stim_period <= StimStart_shift+stim_duration))&& (jj==1)
    I_Stim = stim_amplitude;
else
    I_Stim = 0.0;
end

E_K = R2 * Tem / F * log(K_o / K_i);
i_ss = g_ss * r_ss * s_ss * (V - E_K);
i_B_K = g_B_K * (V - E_K);
i_t = g_t * r1 * (a_to * s + b_to * s_slow) * (V - E_K);

% new i_K1 (from TNNP, fitted last square method)
term1_iK1 = (48.0e-3 / (exp((V + 37.0) / 25.0) + exp((V + 37.0) / -25.0)) + 10.0e-3) * 0.001 / (1.0 + exp((V - (E_K + 76.77)) / -17.0));
alphaK1 = 10.10001681/(1+exp(-0.69561015*(V-E_K-4.148354)));
betaK1 = (5.32060272*exp(0.05012022*(V-E_K+17.51161017))+exp(0.1*(V-E_K-10)))/(1+exp(-0.5*(V-E_K)));
xK1inf = alphaK1/(alphaK1+betaK1);
term2_iK1 = g_K1*xK1inf*((K_o/5.4)^deg_Ko_K1)*(V-E_K);
i_K1 = term1_iK1 + term2_iK1;

% old i_K1
%term1_iK1 = (48.0e-3 / (exp((V + 37.0) / 25.0) + exp((V + 37.0) / -25.0)) + 10.0e-3) * 0.001 \
%       / (1.0 + exp((V - (E_K + 76.77)) / -17.0))
%term2_iK1 = g_K1 * (V - (E_K + 1.73)) / ((1.0 + exp(1.613 * F * (V - (E_K + 1.73)) / (R2 * Tem))) * (
%        1.0 + exp((K_o - 0.9988) / -0.124)))
%i_K1 = term1_iK1 + term2_iK1

i_f_K = g_f * y * f_K * (V - E_K);

% АТФ-зависимый калиевый трансмембранный ток (I_K_ATP)
% P_ATP = 1 / (1 + (self.ATP_i/0.25)^2)  % in Kursanov Protocol
P_ATP = 1 / (1 + (ATP_i / V50_P_ATP) ^ hill_P_ATP) ; % in NICHOLS&LEDERER 1989
i_K_ATP = g_K_ATP * P_ATP * (K_o / K_o_norm) ^ 0.24 * (V - E_K);

dY(10, 1) = - (I_Stim + i_ss + i_B_K + i_t + i_K1 + i_f_K + -2.0 * i_NaK + i_K_ATP) * 1.0 / (V_myo_uL * F);  % K_i

%I_TRPN = GeneralTnC - TRPN;  % свободный тропонин
dY(9, 1) = beta_CMDN * (I_RyR_2 - I_SERCA + I_SR - dY(12, 1) - (-2.0 * I_NaCa_1 + I_pCa_1 + I_CaB_1 + I_LCC_2)/(2.0 * V_myo_uL * F));  % Ca_i !!! dot_TRPN - TRPN - конц. кальций-тропониновых комплексов
dY(8, 1) = V_myo_uL / V_SR_uL * (-I_RyR_2 + I_SERCA - I_SR);  % Ca_SR
i_f = i_f_Na + i_f_K;

dY(13, 1) = -(i_Na + i_t + i_ss + i_f + i_K1 + i_B_Na + i_B_K + i_NaK + I_Stim + I_CaB_1 + I_NaCa_1 + I_pCa_1 + I_LCC_2 + i_K_ATP) / Cm;  % V

h_infinity = 1.0 / (1.0 + exp((V + 76.1) / 6.07));
if (V >= -40.0)
    tau_h = 0.4537 * (1.0 + exp(-(V + 10.66) / 11.1));
else
    tau_h = 3.49 / (0.135 * exp(-(V + 80.0) / 6.8) + 3.56 * exp(0.079 * V) + 310000.0 * exp(0.35 * V));
end
dY(18, 1) = (h_infinity - h) / tau_h;

j_infinity = 1.0 / (1.0 + exp((V + 76.1) / 6.07));
if (V >= -40.0)
    tau_j = 11.63 * (1.0 + exp(-0.1 * (V + 32.0))) / exp(-0.0000002535 * V);
else
    tau_j = 3.49 / ((V + 37.78) / (1.0 + exp(0.311 * (V + 79.23))) * (-127140.0 * exp(0.2444 * V) - 0.00003474 * exp(-0.04391 * V)) + 0.1212 * exp(-0.01052 * V) / (1.0 + exp(-0.1378 * (V + 40.14))));
end
dY(19, 1) = (j_infinity - j_sod) / tau_j;

m_infinity = 1.0 / (1.0 + exp((V + 45.0) / -6.5));
 
if (V - 47.13 <= 1e-5) 
    tau_m = 0.1510178;
else
    tau_m = 1.36 / (0.32 * (V + 47.13) / (1.0 - exp(-0.1 * (V + 47.13))) + 0.08 * exp(-V / 11.0));
end
dY(20, 1) = (m_infinity - m_iNa) / tau_m;

r_ss_infinity = 1.0 / (1.0 + exp((V + 11.5) / -11.82));
tau_r_ss = 10000.0 / (45.16 * exp(0.03577 * (V + 50.0)) + 98.9 * exp(-0.1 * (V + 38.0)));
dY(21, 1) = (r_ss_infinity - r_ss) / tau_r_ss;

s_ss_infinity = 1.0 / (1.0 + exp((V + 87.5) / 10.3));
dY(22, 1) = (s_ss_infinity - s_ss) / tau_s_ss;

% не используются
dY(14, 1) = 0.0;
dY(15, 1) = 0.0;
dY(16, 1) = 0.0;
dY(17, 1) = 0.0;
dY(23, 1) = 0.0;
dY(24, 1) = 0.0;

cell_cur(1) = i_Na;
cell_cur(2) = i_t;
cell_cur(3) = i_ss;
cell_cur(4) = i_f;
cell_cur(5) = i_K1;
cell_cur(6) = i_B_Na;
cell_cur(7) = i_B_K;
cell_cur(8) = i_NaK;
cell_cur(9) = I_Stim;
cell_cur(10) = I_CaB_1;
cell_cur(11) = I_NaCa_1;
cell_cur(12) = I_pCa_1;
cell_cur(13) = I_LCC_2;
cell_cur(14) = i_K_ATP;


%===============================================================================
% End of file
%===============================================================================
