function F = l2l3(x)
alpha_1 = 21;  % per_micrometre (in parameters)
beta_1 = 0.94; % millinewton (in parameters)
alpha_2 = 14.6;  % per_micrometre (in parameters)
beta_2 = 0.0018;  % millinewton (in parameters)
alpha_3 = 33.79;  % per_micrometre (in parameters)
beta_3 = 0.0084;  % millinewton (in parameters)

nondimension = 1;
L_0 = 1.67;
Restlength = L_0;

if nondimension
    alpha_1 = alpha_1*Restlength;
    alpha_2 = alpha_2*Restlength;
    alpha_3 = alpha_3*Restlength;
end
global l1_n n L dx;
F(n+1)=0;
for i=1:n
    F(i) = beta_2*(exp(alpha_2*x(i))-1.0)+beta_1*(exp(alpha_1*(x(i)-l1_n(i)))-1.0)-...
      beta_3*(exp(alpha_3*x(n+1))-1.0);
    F(n+1)=x(i)+F(n+1);
    a=x(1);
    b=x(n);
end

F(n+1)=(F(n+1) -(a/2 + b/2))*dx + x(n+1) - L;