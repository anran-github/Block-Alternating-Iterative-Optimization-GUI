close all
clear all

% CODE REVISED FROM heater_new_model_hardware.m

% ===============SS MODEL and Observer settings===========================
% give canonical form of matrices:
% These values come from "heater_new_model.m": 
A = [0 -0.08091/159; 1 -15.34/159];
B = [0.06545/159; -0.72/159];
C = [0 1];
D = 0;

sys_d=ss(A,B,C,D);

% Discretize the transfer function with Ts = 3 seconds
Ts = 0.5;
sys_ss_d = c2d(sys_d, Ts);
% Get the state-space matrices of the discrete system
[A_d, B_d, C_d, D_d] = ssdata(sys_ss_d);

% Desired observer poles: should be less than unit circle in Discrete-time
observer_poles = [0.85 0.9];

% Place the poles of the observer
L = place(A_d', C_d', observer_poles)';
% eig(A_d-L*C_d)


% =================MULTI-AGENT PARAM SETTINGS===================

% Define LQR parameters
Q = [0.1 0;0 10];  % State cost (penalize deviation of states)
R = 0.001;       % Control effort cost (penalize large control inputs)

% filename for saving dataset
% Compute the LQR controller gain matrix K
K = dlqr(A_d, B_d, Q, R);

x_observer = 0;      % Estimated state (observer)

% try to find out x1_ref
G = inv(C_d * inv(eye(2) - A_d + B_d * K) * B_d);



% saving file setting
file_name = 'data_heater_yalmip_time.csv';

%% DATA COLLECTION SECTION:


for r=5:70 % reference

    error = 100;
    
    % Initial conditions
    x_observer = 0;      % Estimated state (observer)
    k = 1;

    % find x_r with LQR;
    % xr = inv(eye(2) - A_d + B_d * K) * B_d*G*r;
    xr=[0;r];
    
    % continue iterations until error reached.
    cache_matrix = zeros(2,4);
    while error > 0.01 
    
        if k == 1
            xt = [x_observer;0];
        end
        % Select different control law for the discrete-time system: 
        % [u,P,theta] = multi_agent_algorithm(xt,xr,A_d,B_d,R,Q,K,G)
        [u,P,theta] = heater_original_optimation(xt,xr,A_d, B_d,R,Q,K,G);

        % ((A_d*xt+B_d*u-xr)'*p_init*(A_d*xt+B_d*u-xr))^(0.5)-((xt-xr)'*p_init*(xt-xr))^(0.5)<=-0.0001*((xt-xr)'*(xt-xr))^(0.5)

        % will not consider saturate for data collection. test only
        % if u>=100
        %     u=100;
        % elseif u<=0
        %     u=0;
        % end
    
        % update observer
        y = C_d * xt;  
        if k == 1
            x_observer = A_d * xt + B_d * u + L * (y - C_d * xt);  % Observer update
        else
            x_observer = A_d * x_hat_store(:,k-1) + B_d * u + L * (y - C_d * x_hat_store(:,k-1));  % Observer update
        end
    
        x_hat_store(:, k) = x_observer;
        
        % Store results
        % ========== Data Format ===========
        %  x1      p1   p2   u
        %  x2(Th)  p2   p3   reference
        % ==================================

        % SAVE RESULT
        cache_matrix(:,1) = xt;
        % eig(p)
        cache_matrix(:,2:3) = P;
        cache_matrix(1,4) = u;
        cache_matrix(2,4) = r;
        if k==1
            result_matrix = cache_matrix;
        else
            result_matrix = cat(1,cache_matrix,result_matrix);
        end
        
        % update temperature with SS model.
        x_2 = A_d * xt + B_d * u;   % Adx+Bdu for discrete-time 
        xt = [x_observer(1);x_2(2)];

        error = abs(x_2(2) - r);
        k = k+1;
    end

    % save data
    writematrix(result_matrix, file_name,'WriteMode','append');
    clear result_matrix x_hat_store 

end
