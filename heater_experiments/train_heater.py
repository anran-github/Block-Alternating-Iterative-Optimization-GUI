import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse
import ast


from network import P_Net


class Data_Purify():
    def __init__(self,filename,xr,p_max,u_max) -> None:

        # Load the CSV dataset
        df = pd.read_csv(filename, header=None)

        # Format:
        # x1 p1 p2 u
        # x2 p2 p3 theta
        self.purify_factor_p = p_max
        self.purify_factor_u = u_max

        # Assuming the first row is the header, adjust if needed
        data = df.values  # Transpose to have shape (2, n)


        self.data_x = []
        self.data_u = []
        self.data_p = []
        self.data_r = []
        instance = 2
        for row in range(data.shape[0]//instance):
            self.data_x.extend([data[instance*row:instance*row+instance,0]])
            self.data_u.extend([data[instance*row,3]])
            self.data_p.extend([data[instance*row:instance*row+instance,1:3]])
            self.data_r.extend([data[instance*row+1,3]])

        
        # self.data_r = np.tile(xr,(len(self.data_u),1))
        print('----------DATA SUMMARY------------')
        print(f'There are {len(self.data_u)} raw data points')

        # combine [x1,x2] and r as a single input:
        self.input_data_combine = [np.append(self.data_x[i],self.data_r[i]) for i in range(len(self.data_x))]

        # now we only need 4: p1,p2,p3,..., p10, u
        self.label_data = [[self.data_p[i][0,0],self.data_p[i][0,1],self.data_p[i][1,1],
                            self.data_u[i]] for i in range(len(self.data_u))]

    def return_data(self,):
            return self.input_data_combine, self.label_data



    def purified_data(self):

        data_p1 = [x[0,0] for x in self.data_p]
        data_p2 = [x[0,1] for x in self.data_p]
        data_p3 = [x[1,1] for x in self.data_p]

        data_p1 = np.array(data_p1)
        data_p2 = np.array(data_p2)
        data_p3 = np.array(data_p3)

        self.data_u = np.array(self.data_u)
        mask_outlier1 = np.abs(data_p1) < (self.purify_factor_p)
        mask_outlier3 = np.abs(data_p2) < (self.purify_factor_p)
        mask_outlier4 = np.abs(data_p3) < (self.purify_factor_p)
        mask_outlier2 = np.abs(self.data_u) < (self.purify_factor_u)
        self.mask_outlier = np.logical_and(
            np.logical_and(mask_outlier1,mask_outlier2),
            np.logical_and(mask_outlier3,mask_outlier4)
        )

        # for outliers
        self.outlier_mask = np.logical_not(self.mask_outlier)
        self.outliers = np.array(self.data_x)[self.outlier_mask]
        # use mask get rid of outliers
        self.data_p = np.array(self.data_p)[self.mask_outlier]
        self.data_r = np.array(self.data_r)[self.mask_outlier]
        self.data_x = np.array(self.data_x)[self.mask_outlier]
        self.data_u = np.array(self.data_u)[self.mask_outlier]
        print(f'Dataset is purified! Now there are {len(self.data_r)} data points available.')
        print('--------------------------------')

        self.len_data = len(self.data_u)

        # Final Data for feeding NN.

        # combine [x1,x2] and r as a single input:
        self.input_data_combine = [np.append(self.data_x[i],self.data_r[i]) for i in range(len(self.data_x))]

        # now we only need 4: p1,p2,p3,u
        self.label_data = [[self.data_p[i][0,0],self.data_p[i][0,1],self.data_p[i][1,1],self.data_u[i]] for i in range(len(self.data_u))]

        return self.input_data_combine, self.label_data


    def draw_data(self, data_u, data_p):

        plt.subplot(221)
        plt.plot(list(range(len(data_u))),data_u)
        # plt.show()
        plt.title('Parameter u(t)')
        # plt.close()

        plt.subplot(222)
        data_p1 = [x[0,0] for x in data_p]
        plt.plot(list(range(len(data_p1))),data_p1)
        # plt.show()
        plt.title('Parameter p(1)')

        plt.subplot(223)
        data_p2 = [x[0,1] for x in data_p]
        plt.plot(list(range(len(data_p2))),data_p2)
        # plt.show()
        plt.title('Parameter p2')

        plt.subplot(224)
        data_p3 = [x[1,1] for x in data_p]
        plt.plot(list(range(len(data_p3))),data_p3)
        plt.show()
        plt.title('Parameter p3')
        plt.close()




# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")


# Parse the command-line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='data_heater_yalmip.csv', help='corresponding theta dataset')
parser.add_argument('--reference',type=float,default=0, help='reference of dataset')
parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
parser.add_argument('--pre_trained', type=str, default='',help='input your pretrained weight path if you want')
args = parser.parse_args()
print(args)

save_path = f'weights'
if not os.path.exists(save_path):
    os.makedirs(save_path)

x_r = np.array([args.reference])
data_tmp = Data_Purify(args.dataset,x_r,p_max=2000,u_max=1e5)




# import dataset 
input_data_valid,label_data_valid = data_tmp.purified_data()
data_tmp.draw_data(data_tmp.data_u,data_tmp.data_p)



# ==============DATA FILTER===============
# Filter with std and mean: For p1
# std,d_mean = torch.std_mean(label_data[:,1])    
# mask = label_data[:,1]<=(d_mean+std)
# input_data_valid = input_data[mask]
# label_data_valid = label_data[mask]

# ===========STD-MEAN Norm Output Label================
# std,mean = torch.std_mean(label_data_valid,dim=0)
# print('===========STD AND MEAN OF DATASET================')
# print(f'STD: {std},\nMEAN: {mean}')
# print('==================================================')
# label_normalized = (label_data_valid-mean)/(std+1e-10)

# ===========MIN-MAX Norm Output Label================
# d_min,_ = torch.min(label_data_valid,dim=0)
# d_max,_ = torch.max(label_data_valid,dim=0)
# print('===========MIN and MAX OF DATASET================')
# print(f'MIN: {d_min},\n MAX: {d_max}')
# print('==================================================')
# label_normalized = (label_data_valid-d_min)/(d_max-d_min+1e-10)


X_train, X_test, y_train, y_test = train_test_split(input_data_valid, label_data_valid, test_size=0.2, random_state=42)

X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 1024
num_epochs = 30000

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





model = P_Net(output_size=4).to(device)

if len(args.pre_trained):
    model.load_state_dict(torch.load(args.pre_trained))
    model.eval()
    print('----------added previous weights: {}------------'.format(args.pre_trained))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs//100, eta_min=args.lr*0.07)




global lowest_loss 
lowest_loss = np.inf

def test(model, test_loader,epoch):
    test_loss = 0.0
    u_losses = 0
    model.eval()
    with torch.no_grad():
        loop = tqdm(test_loader)
        for inputs, targets in loop:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            u_loss = criterion(outputs[:,3], targets[:,3].to(device))
            # print(outputs)
            # print(targets)
            # print(loss)
            if not 'cuda' in device.type:
                test_loss += loss.item() * inputs.size(0)
                u_losses  += u_loss.item()
            else:
                test_loss += loss.cpu().item() * inputs.size(0)
                u_losses  += u_loss.cpu().item()

            # loop.set_postfix(loss=f"{loss_set[-1]:.4f}", refresh=True)
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

    # save model
    # Save the model
    global lowest_loss
    if lowest_loss>=test_loss:
        lowest_loss = test_loss
        print('Model Saved!')
        torch.save(model.state_dict(), os.path.join(
            save_path,'new_model_epoch{}_{:.3f}_u{:.3f}.pth'.format(
                epoch,lowest_loss,u_losses)))




test(model, test_loader,epoch=0)


losses = []
loss_avg = []
model.train()
for epoch in range(30000,int(30000+num_epochs)):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss1 = 0.1*criterion(outputs[:,:3], targets[:,:3].to(device))
        loss2 = 10*criterion(outputs[:,3], targets[:,3].to(device))
        # loss1 = criterion(outputs[:,:3], targets[:,:3].to(device))
        # loss2 = 10*criterion(outputs[:,3], targets[:,3].to(device))
        loss = loss1 + loss2
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100)
    # update lr and gradient
    optimizer.step()
    scheduler.step()

    loss_avg.append(loss.item())  # Store the loss value for plotting

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    losses.append(np.average(loss_avg))

    if epoch >= 100 and epoch%50==0:
        test(model, test_loader,epoch)

# Plot the loss dynamically
plt.clf()  # Clear previous plot
plt.plot(losses, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
# plt.pause(0.05)  # Pause for a short time to update the plot
plt.savefig(os.path.join(save_path,'training_loss_{}.png'.format(num_epochs)))
plt.plot()




