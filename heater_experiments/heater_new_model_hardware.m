close all
clear all

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
ambient_t = arduino_lab1(0);
desired_t = 30;
x_observer = 0;      % Estimated state (observer)

G = inv(C_d * inv(eye(2) - A_d + B_d * K) * B_d);
% ref = inv(eye(2) - A_d + B_d * K) * B_d*G*desired_t
% xr = inv(eye(2) - A_d + B_d * K) * B_d*G*r;

xr=[0;30];

% ===================LQR END========================


% ================== NN SETTINGS =========================
% Load ONNX model
net = importONNXNetwork("heater_NN2.onnx", 'InputDataFormats', {'BC'}, 'OutputDataFormats', {'BC'});

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
    % u = -K * xt +G*desired_t;

    % YALMIP
    % [u,p_init,theta_init] = heater_original_optimation(xt,xr,A_d, B_d,R,Q,K,G);
    
    % NN MODEL
    data_input = [xt(1),xt(2),xr(2)];
    output = predict(net, data_input);
    u = output(4)


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

    time_rest = Ts - elapsedTime;
    if time_rest > 0
        pause(time_rest-0.028);
    end

    x_2 = arduino_lab1(u)-ambient_t  % read actural temperature.
    xt = [x_observer(1);x_2];

    % Store results
    x_store(:, k) = xt;
    u_store(k) = u;

    total_time = toc;


    disp(['Computing time: ', num2str(elapsedTime), ' seconds; ', ...
        'Total Elapsed time: ', num2str(total_time), ' seconds']);

end

save('result30_NN_new2.mat')
%% Plot the results--SHOW UP IN PAPER

% TO DISPLAY PAPER FIGURE: LOAD SAVED .mat DATA. uncomment following:
clear
load("result30_NN_yalmip_10240.mat");
x_store_lqr = x_store;
u_store_lqr = u_store;

load("result30_NN_new2.mat")
time = (0:N-1) * Ts;

figure;
% subplot(3, 1, 1);
% plot(time, x_store(1, :), 'r', 'LineWidth', 2);
% hold on
% plot(time, x_store_lqr(1, :), 'b', 'LineWidth', 2);
% xlabel('Time [s]');
% ylabel('State x_1');
% legend('Multi-Agent', 'YALMIP');
% grid("on");
% title('State x_1');

subplot(2, 1, 1);
plot(time, x_store(2, :)+20, 'b', 'LineWidth', 2);
hold on;
% plot(time, x_hat_store(2, :), 'r--', 'LineWidth', 2);
% plot reference
plot(time, x_store_lqr(2, :)+20, 'r', 'LineWidth', 2);
plot(time,ones(1,size(time,2))*xr(2)+20,'c--', 'LineWidth', 2);
ylim([20 53]);
xlabel('Time [s]');
ylabel('Temperature');
legend('Proposed Method', 'Non-Optimized Method','Reference',Location='southeast');
grid("on");

subplot(2, 1, 2);
plot(time, u_store_lqr, 'm', 'LineWidth', 2);
hold on
plot(time, u_store, 'k', 'LineWidth', 2);

legend('Non-Optimized Method','Proposed Method');
grid("on");
xlabel('Time [s]');
ylabel('Control Input');


% exportgraphics(gcf, 'heater_res.png', 'Resolution', 300);
