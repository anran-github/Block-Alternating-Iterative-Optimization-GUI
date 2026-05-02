'''
This file draws:

===============================================
These figures will NOT be inserted into final paper.
===============================================
Test trained NOM NN:
1. Given r=0 and one start point, plot figures of x1 and x2 change vs time.


==============================================================
Summary of Errors between xi and x_r with 10000 random points.
==============================================================
'''


import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import argparse
from scipy.signal import cont2discrete, place_poles
# from control import place
# from network_ori import P_Net
from network import P_Net



# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"Using device: {device}")


# Objective function setting
parser = argparse.ArgumentParser(description="Process some input parameters.")
parser.add_argument('--theta', type=float, default=0.01, help="Theta: [0.1, 0.01, 0.001]")
parser.add_argument('--resume', type=bool, default=False, help="Reference: range between [-1,1]")
args = parser.parse_args()



# % read A,B,C,D matrices:
A = np.array([[0, -0.08091/159], [1, -15.34/159]])
B = np.array([[0.06545/159], [-0.72/159]])
C = np.array([0, 1])
D = 0

# Define the sample time (Ts)
Ts = 0.5  # Adjust the sample time as needed
# Discretize the system
Ad, Bd, Cd, _, _ = cont2discrete((A, B, C, D), Ts, method='zoh')
Cd = Cd.reshape(1,2)
# Ad = torch.tensor(Ad,dtype=torch.float32)
# Bd = torch.tensor(Bd,dtype=torch.float32)
T_ambient = 22
reference = 50
x_input = [0,0,reference]

# build observer gains:
observer_poles = np.array([0.85, 0.9])
L_res = place_poles(Ad.T,Cd.T,observer_poles)
L = (L_res.gain_matrix).T



x_r = torch.tensor([0.,reference]).reshape(2,1).to(device)
theta=torch.tensor(args.theta).to(device)





model = P_Net(output_size=4).to(device)
print(model)
# weights = glob('weights/*.pth')
weights = ['weights/new_model_epoch43500_12680.934_u5.884.pth',

]
weights.sort()

for weight in weights:
    print(weight)
    model.load_state_dict(torch.load(weight,map_location=device))

    # Set the model to evaluation mode (if needed)
    model.eval()


    # give a random point, plot how it goes after n iterations
    num_points = 1
    iteration = 500
    u_set = []
    x1_set = []
    x2_set = []
    delta_x_set = []
    v_dot_set = []
    arrow_trajectory = []

    # Bd=np.array([0,0.1]).reshape(2,1)
    # Ad=lambda xt: [[1, 0.1],[-0.1*9.8*np.cos(xt), 1]]



    for t in range(iteration):

        if t == 0:
            xt = np.array([0.,0]).reshape(2,1)
        # p_tt = np.eye(2)

        x_input = [xt[0,0],xt[1,0],reference]
        with torch.no_grad():
            x_tem = torch.tensor(x_input).reshape(1,3)
            x_tem = x_tem.type(torch.float32)
            # output order: p1, p2, p3, u
            output = model(x_tem.to(device))
            if not device.type == 'cuda':
                output = output.numpy()
            else:
                output = output.detach().cpu().numpy()

            # k, _, _ = dlqr(A, B, Q, R)
            # Ad= np.array([np.array([1, 0.1]),[-0.1*9.8*np.cos(xt.numpy()), 1]])
            # Ad= np.stack([xt[:,0]+0.1*xt[:,1],-0.1*9.8*np.sin(xt[:,0]) + 1*xt[:,1]]).transpose(1,0,2)

            # p_t = np.array([[output[0,0],output[0,1]],[output[0,1],output[0,2]]])
            # print(output)
            u_t = np.array(output[:,3]).reshape(1,1)
            print(output)
            # saturate input u
            u_value = u_t[0]
            if u_value > 100:
                u_t = u_t*0+100
            elif u_value < 0:
                u_t = u_t*0

            #  observer update
            y = Cd @ xt;  
            x_observer = Ad @ xt + Bd @ u_t + L @ (y - Cd @ xt)  
                    
            

            # uncomment this line if you want to compare to dlqr.
            # u_t = -torch.from_numpy(k).type(torch.float32) @ xt
            xtt =  Ad @ xt + Bd @ u_t

            # update x(t+1)
            xtt[0,0] = x_observer[0,0]

            x1_set.append(xt[0,0])
            x2_set.append(xt[1,0])
            u_set.append(u_t[0,0])
            # our previous V

            # v_dot = np.sqrt(np.linalg.norm(xtt.T @ p_t @ xtt ,axis=1)) - np.sqrt(np.linalg.norm(xt.T @ p_tt @ xt,axis=1))
            # v_dot_set.append(v_dot)
            # u_set.append(u_t.item())
            ##======REMEMBER TO CHEKCK THETA HERE (0.01)======
            # arrow_trajectory.append((-0.01*np.linalg.norm(xt)))
            delta_x_set.append((xtt-xt))
            xt = xtt
            # p_tt = p_t


    # transfer to degrees
    delta_x_set = np.array(delta_x_set)
    u_set = np.array(u_set)
    x1_set = np.array(x1_set)
    x2_set = np.array(x2_set)


    # arrow_trajectory = np.array(arrow_trajectory)*180/np.pi



    # =================================================
    # Plot Results: x1-t, x2-t, u-t, Delta V - t, x1-x2
    # =================================================
    # introduce latex
    plt.rcParams['text.usetex'] = True

    # control number font size on axis.
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["font.family"] = "Times New Roman"


    # control number font size on axis.
    plt.rcParams.clear()
    plt.rcParams.update()

    t = Ts*np.array(list(range(iteration)))
    plt.subplot(311)
    plt.plot(t,x1_set)
    plt.grid(linestyle = '--')
    plt.xlabel(r'Time (s)', fontsize=16)
    plt.ylabel(r'$x_1$ (observer)', fontsize=16)

    plt.subplot(312)
    plt.plot(t,x2_set,label='temperature')
    plt.plot(t,np.ones(len(range(iteration)))*reference,'--r',label='reference')
    plt.grid(linestyle = '--')
    # plt.legend()
    # plt.legend('temperature','reference')
    plt.xlabel(r'Time (s)', fontsize=16)
    plt.ylabel(r'$Temp$ (deg)', fontsize=16)

    plt.subplot(313)
    plt.plot(t,u_set)
    plt.grid(linestyle = '--')
    plt.xlabel(r'Time (s)', fontsize=16)
    plt.ylabel(r'U', fontsize=16)

    plt.tight_layout()
    # plt.savefig('paper_figures/x1x2.png')
    plt.show()
    plt.close()


# this code comes from drones
'''
    fig, ax = plt.subplots()

    for i in range(len(x1_set)):
        
        # arrow_trajectory = np.array(delta_x_set)*180/np.pi
        ax.quiver(x1_set[i], x2_set[i],delta_x_set[i,0,0], delta_x_set[i,1,0],
                color="C0", angles='xy',scale_units='xy', scale=2, width=.006,headaxislength=4)


    plt.grid(linestyle = '--')
    plt.xlabel(r'$x_1$ (deg)', fontsize=16)
    plt.ylabel(r'$x_2$ (deg/s)', fontsize=16)


    # plt.title('input data changes with random points')
    plt.tight_layout()
    # plt.savefig('paper_figures/x1x2.png')
    plt.show()
    plt.close()
'''
