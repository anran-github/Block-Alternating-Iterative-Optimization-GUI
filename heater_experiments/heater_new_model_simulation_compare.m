close all
clear all

% ==========================================
%         Heater System Simulation
% Note: The simulation results are a little
%       different from hardware experiments
% 
% Given control laws: LQR, Multi-Agent,
%                     Neural Controller
% ==========================================


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


% =================LQR SETTINGS===================

% Define LQR parameters
Q = [0.1 0;0 10];  % State cost (penalize deviation of states)
R = 0.001;       % Control effort cost (penalize large control inputs)

% Compute the LQR controller gain matrix K
K = dlqr(A_d, B_d, Q, R);

% Initial conditions
ambient_t = 20;
desired_t = 30;
x_observer = 0;      % Estimated state (observer)

G = inv(C_d * inv(eye(2) - A_d + B_d * K) * B_d);
% ref = inv(eye(2) - A_d + B_d * K) * B_d*G*desired_t
% xr = inv(eye(2) - A_d + B_d * K) * B_d*G*r;

xr=[0;30];

% ===================LQR END========================


% ================== NN SETTINGS =========================
% Load ONNX model
% net = importONNXNetwork("heater_NN2.onnx", 'InputDataFormats', {'BC'}, 'OutputDataFormats', {'BC'});

% ==================NN setting end =========================


% Simulation parameters
T_final = 300;    % Final simulation time
N = T_final / Ts;  % Number of discrete time steps

% Arrays to store simulation results
x_store = zeros(2, N);        % Store actual state
x_hat_store = zeros(2, N);    % Store estimated state
u_store = zeros(1, N);        % Store control input


%% CORE SECTION:

for k = 1:N

    tic;

    if k == 1
        xt = [x_observer;0];
    end

    % different control laws for discrete-time system: 
    
    % LQR
    u = -K * xt +G*desired_t;

    % Mutil-Agent: time-consuming
    % [u,P,theta] = multi_agent_algorithm(xt,xr,A_d,B_d,R,Q,K,G)
    
    % NN MODEL
    % data_input = [xt(1),xt(2),xr(2)];
    % output = predict(net, data_input);
    % u = output(4);


    if u > 100
        u = 100;
    elseif u < 0
        u = 0;
    end


    y = C_d * xt;  % Measurement
    if k == 1
        x_observer = A_d * xt + B_d * u + L * (y - C_d * xt);  % Observer update
    else
        x_observer = A_d * x_hat_store(:,k-1) + B_d * u + L * (y - C_d * x_hat_store(:,k-1));  % Observer update
    end
    x_hat_store(:, k) = x_observer;


    elapsedTime = toc;  % Stop the timer and return the elapsed time


    % x_2 = arduino_lab1(u)-ambient_t  % read actural temperature.
    xt = [x_observer(1);x_observer(2)];

    % Store results
    x_store(:, k) = xt;
    u_store(k) = u;

    total_time = toc;


    disp(['Computing time: ', num2str(elapsedTime), ' seconds; ', ...
        'Total Elapsed time: ', num2str(total_time), ' seconds']);

end


%% Plot the results

time = (0:N-1) * Ts;

figure;
subplot(3, 1, 1);
plot(time, x_store(1, :), 'r', 'LineWidth', 2);
hold on
xlabel('Time [s]');
ylabel('State x_1');
legend('Multi-Agent');
grid("on");
title('State x_1');

subplot(3, 1, 2);
plot(time, x_store(2, :)+20, 'r', 'LineWidth', 2);
hold on;
% plot(time, x_hat_store(2, :), 'r--', 'LineWidth', 2);
% plot reference
plot(time,ones(1,size(time,2))*xr(2)+20,'c--', 'LineWidth', 2);
xlabel('Time [s]');
ylabel('State x_2');
legend('Multi-Agent','Reference');
grid("on");
title('State x_2');

subplot(3, 1, 3);
plot(time, u_store, 'k', 'LineWidth', 2);

legend('LQR','Multi-Agent');
grid("on");
xlabel('Time [s]');
ylabel('Control input u');
title('Control input u');
% exportgraphics(gcf, 'heater_res.png', 'Resolution', 300);