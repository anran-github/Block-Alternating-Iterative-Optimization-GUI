clc
clear

A = [0 1; 0 0];
B = [0; 1];
C=[1 0];
D=0;

G=ss(A,B,C,D);

Gd=c2d(G,0.1);
Ad=Gd.A;
Bd=Gd.B;

% Load ONNX model
net = importNetworkFromONNX("super_resolution.onnx")
% model = importONNXNetwork('model.onnx');

net.Layers(1).InputInformation


X = dlarray(rand(1, 3), 'UU');
net = initialize(net, X);
summary(net)



example_x = dlarray([-0.5,0,0], 'UU');


% Perform inference
output = predict(net, example_x);

% Display the output (adjust as per your model's output)
disp('Model Output:');
disp(output);


