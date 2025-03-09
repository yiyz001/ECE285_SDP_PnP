
% Solve the PnP problem using SOS hierarchy.
% Notations are the same as the proposal and slides.
    
% Define the problem parameters
Omega = zeros(9,9);
n = 10;
m = randn(3,n);
u = rand(3,n);
A = zeros(3,9,n);
Q = zeros(3,3,n);
I_9 = eye(9);
I_3 = eye(3);
e_3 = [0 0 1]';

sum_Q = zeros(3,3);
sum_QA = zeros(3,9);

for i = 1:n
    A(:,:,i) = blkdiag(m(:,i)',m(:,i)',m(:,i)');
    Q(:,:,i) = (u(:,i)*e_3' - I_3)' * (u(:,i)*e_3' - I_3);
    sum_Q = sum_Q + Q(:,:,i);
    sum_QA = sum_QA + Q(:,:,i)*A(:,:,i);
end

P = sum_Q \ sum_QA;

for i = 1:n
    Omega = Omega + (A(:,:,i)+P)' * Q(:,:,i) * (A(:,:,i)+P);
end

% Define constraints
Vs = {}; % V_j matrices
Vs{1} = E_ij(1:3,1:3);
Vs{2} = E_ij(4:6,4:6);
Vs{3} = 0.5 * (E_ij(1:3,4:6) + E_ij(4:6,1:3));
Vs{4} = 0.5 * (E_ij(1:3,7:9) + E_ij(7:9,1:3));
Vs{5} = 0.5 * (E_ij(4:6,7:9) + E_ij(7:9,4:6));
Vs{6} = 0.5 * (E_ij(2,6) + E_ij(6,2) - E_ij(3,5) - E_ij(5,3));
Vs{7} = 0.5 * (E_ij(3,4) + E_ij(4,3) - E_ij(1,6) - E_ij(6,1));
Vs{8} = 0.5 * (E_ij(1,5) + E_ij(5,1) - E_ij(2,4) - E_ij(4,2));

vs = {}; % v_j vectors 
vs{1} = zeros(9,1);
vs{2} = zeros(9,1);
vs{3} = zeros(9,1);
vs{4} = zeros(9,1);
vs{5} = zeros(9,1);
vs{6} = -I_9(:,7);
vs{7} = -I_9(:,8);
vs{8} = -I_9(:,9);

cs = {}; % constants c_j
cs{1} = -1;
cs{2} = -1;
cs{3} = 0;
cs{4} = 0;
cs{5} = 0;
cs{6} = 0;
cs{7} = 0;
cs{8}= 0;
%% 


r = 2; % hierarchy number
eps= 0.1; % softening parameter (when it is zero, the formulation is exact.)
gamma_r = []; % list of gamma at each hierarchy

for i = 1:r
    yalmip('clear')
    x = sdpvar(9,1);
    gamma = sdpvar(1,1);
    p = x'*Omega*x;
    
    s_plus= {};
    s_minus= {};

    Constraints = [];
    p_sos = p - gamma; 
    for j = 1:8
        s_p = polynomial(x,2*(i-1) );
        s_m = polynomial(x,2*(i-1) );
        Constraints = [Constraints, sos(s_p), sos(s_m)];

        s_plus{j} = s_p;
        s_minus{j} = s_m;

        p_sos = p_sos + (s_p-s_m)*g_j(Vs{j},vs{j},cs{j},x) + eps*(s_p + s_m);
    end

    Constraints = [Constraints, sos( p_sos )];
    solvesos(Constraints,-gamma,[],[x;gamma])
    value(gamma)
    gamma_r = [gamma_r,value(gamma)];
end

function output = E_ij(i_range,j_range)
    size_nonzero = length(i_range);
    output = zeros(9,9);
    output(i_range,j_range) = eye(size_nonzero);
end

function output = g_j(V,v,c,x)
    output = x'*V*x + v'*x + c;
end