function F=v_meh(v)

alpha_G = 1.0;   % dimensionless (in parameters_izakov_et_al_1991)
a = 0.25;   % dimensionless (in parameters)
alpha_P = 4.0;   % dimensionless (in parameters)
d_h = 0.5;   % dimensionless (in parameters)
llambda = 55.0;  % millinewton (in parameters)

alpha_2 = 14.6;  % per_micrometre (in parameters) 14.6
beta_2 = 0.0018;  % millinewton (in parameters) 0.0018
alpha_3 = 33.79;  % per_micrometre (in parameters)
beta_3 = 0.0084;  % millinewton (in parameters)

v_max = 0.0055;  %  micrometre_per_second (in parameters)

alpha_vp_l = 16.0;  % per_micrometre (in CE_velocity)
alpha_vp_s = 8;  % per_micrometre (in CE_velocity)
beta_vp_l = 0.00084 * 0.9;  % millinewton_second_per_micrometre (in CE_velocity)
beta_vp_s = 0.84;  % millinewton_second_per_micrometre (in CE_velocity)

%-------------------------------------------------------------------------------
% Cell parameters 
%-------------------------------------------------------------------------------

nondimension = 1;
L_0 = 1.67;
R_0 = 1.05;
Restlength = L_0;

if nondimension
    alpha_2 = alpha_2*Restlength;
    alpha_3 = alpha_3*Restlength;
    alpha_vp_l = alpha_vp_l*Restlength;
    alpha_vp_s = alpha_vp_s*Restlength;
    v_max = v_max/Restlength;

end    

global Lam_mech l1 l2 l3 N_meh IschemiaDeg;

%Here we change ischemia parameters
    %Normal and ischemia zones

if (IschemiaDeg>0)
    llambda = Lam_mech;
end


if (v <= 0.0)
    k_P_vis = beta_vp_l*exp(alpha_vp_l*l1);
else
    k_P_vis = beta_vp_s*exp(alpha_vp_s*l1);
end

v_1 = v_max/10.0;
gamma2 = a*d_h*(0.1)^2.0/(3.0*a*d_h-(a+1.0)*0.1);

if (v <= 0.0)
   P_star = a*(1.0+v/v_max)/(a-v/v_max);
else
   P_star = 1.0+d_h-d_h^2.0*a/(a*d_h/gamma2*(v/v_max)^2.0+(a+1.0)*v/v_max+a*d_h);
end

if (v <= 0.0)
   G_star = 1.0+0.6*v/v_max;
elseif (v <= 0.1)
   G_star = P_star/((0.4*a+1.0)*v/(a*v_max)+1.0);
else
   G_star = P_star*exp(-alpha_G*((v-v_1)/v_max)^alpha_P)/((0.4*a+1.0)*v/(a*v_max)+1.0);
end

if (v <= -v_max)
    p_ext = 0.0;
elseif (v <= 0.0)
    p_ext = a*(1.0+v/v_max)/((a-v/v_max)*(1.0+0.6*v/v_max));
elseif (v <= v_1)
    p_ext = (0.4*a+1.0)*v/(a*v_max)+1;
else
    p_ext = ((0.4*a+1.0)*v/(a*v_max)+1.0)*exp(alpha_G*(v/v_max-0.1)^alpha_P);
end
   
p_v = p_ext;

F=beta_2*(exp(alpha_2*l2)-1.0) + llambda*p_v*N_meh + ...
    k_P_vis*v - beta_3*(exp(alpha_3*l3)-1.0);