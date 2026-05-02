import torch
import torch.onnx
import numpy as np

from network import P_Net




model = P_Net(output_size=4)

# test saved model
# Create an instance of the model
# Load the saved model's state dictionary
# model.load_state_dict(torch.load('trained_model_theta0.01_0.21945167709165908.pth'))
model_path = 'weights/new_model_epoch43500_12680.934_u5.884.pth'
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

# Set the model to evaluation mode (if needed)
model.eval()


# Define example inputs (you may need to adjust the shape and dtype)
# range for x and y direction: x1 [-1,1], x2 [-1,1]
# x1 = 2 * np.random.rand() - 1
# x2 = 2 * np.random.rand() - 1
# r = np.random.rand() - 0.1

# z direction
x = np.random.rand(1,3) 

x0 = torch.tensor(x,dtype=torch.float32)
# Define example inputs (you may need to adjust the shape and dtype)
# example_input = torch.randn(1, input_channels, height, width).to(torch.float32)



# Export the model to ONNX
# torch.onnx.export(model, example_input, 'model.onnx', opset_version=12)


# Export the model
save_name = model_path.split('/')[-1].split('.')[0]
torch.onnx.export(model,               # model being run
                x0,                         # model input (or a tuple for multiple inputs)
                f"{save_name}_yalmip_new.onnx",   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

print('Done!')




# def Drone_onnx_model():

#     from semi_supervison.network import P_Net
#     model = P_Net(output_size=10)

#     # test saved model
#     # Create an instance of the model
#     # Load the saved model's state dictionary
#     # model.load_state_dict(torch.load('trained_model_theta0.01_0.21945167709165908.pth'))
#     model.load_state_dict(torch.load('semi_supervison/DroneZ_MPC_weights/dense_center_S40_vshape_Iter20_Epoch10.pth'))

#     # Set the model to evaluation mode (if needed)
#     model.eval()


#     # Define example inputs (you may need to adjust the shape and dtype)
#     # range for x and y direction: x1 [-1,1], x2 [-1,1]
#     x1 = 2 * np.random.rand() + 0.5
#     x2 = 2 * np.random.rand() - 1
#     r = np.random.rand() + 1

#     x0 = torch.tensor([[x1],[x2],[r]]).reshape(1,3)
#     # Define example inputs (you may need to adjust the shape and dtype)
#     # example_input = torch.randn(1, input_channels, height, width).to(torch.float32)



#     # Export the model to ONNX
#     # torch.onnx.export(model, example_input, 'model.onnx', opset_version=12)


#     # Export the model
#     torch.onnx.export(model,               # model being run
#                     x0,                         # model input (or a tuple for multiple inputs)
#                     "semi_supervison/DroneZ_MPC_weights/dense_center_S40_vshape_Iter20_Epoch10.onnx",   # where to save the model (can be a file or file-like object)
#                     export_params=True,        # store the trained parameter weights inside the model file
#                     opset_version=10,          # the ONNX version to export the model to
#                     do_constant_folding=True,  # whether to execute constant folding for optimization
#                     input_names = ['input'],   # the model's input names
#                     output_names = ['output'], # the model's output names
#                     dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                                     'output' : {0 : 'batch_size'}})

#     print('Done!')


