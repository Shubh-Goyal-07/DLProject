import torch
import torch.optim as optim

import matplotlib.pyplot as plt

import torch.nn as nn

import copy

import os
os.chdir('/home/raid/Desktop/Shubh/DLProject/experiments/')

from optimizers.customAdam import customAdam
from optimizers.customAdam2 import customAdam2
from optimizers.customAdam3 import customAdam3


optimizers_dict = {
                    "Adam_torch" : optim.Adam,}
                    # "RMS_torch" : optim.RMSprop,
                    # "AdaGrad_torch" : optim.Adagrad }


def initialize_weights_xavier(model):
    # torch.manual_seed(0)

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def set_lr(optimizer,factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Helper class user for training and testing
class TrainerAll2():
    def __init__(self, train_loader, test_loader, criterion, epochs=50, lr=0.001, threshold=2):
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = criterion
        self.epochs = epochs
        self.threshold = threshold

        self.lr = lr
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loss_log = {}
        self.train_acc_log = {}

        self.test_loss_log = {}
        self.test_acc_log = {}

    def train(self, model_local, opt):
        # defining local variables for the class
        train_loss_optim = []
        train_acc_optim = []
        test_loss_optim = []
        test_acc_optim = []

        model_local = copy.deepcopy(model_local)
        optimizer = optimizers_dict[opt](model_local.parameters(), self.lr)

        super_epochs = 4  #hyper-parameter
        cnt = 0
        cnt_inc = 0
        cnt_dec = 0
        flag_inc = False
        flag_dec = False
        accumulated_loss = []
        train_acc = 0
        # threshold = 2
        for epoch in range(self.epochs):
            print("Epoch: ",epoch+1)
            train_loss = 0
            model_local.train()
            # super_epochs += 1
            if (not flag_inc) and (not flag_dec):
                cnt += 1

            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model_local(data)
                loss = self.criterion(output, target)
                loss.backward(create_graph = True)
                optimizer.step()
                
                train_loss += loss.item()
                train_acc = 100*torch.sum(torch.argmax(output, dim=1) == target).item()/len(target)
            
            accumulated_loss.append(train_loss)
            if (flag_inc):
                if (cnt_inc!=0):
                    set_lr(optimizer,2)
                    print(f'Learning rate increased to {get_lr(optimizer)} : {cnt_inc}')
                    cnt_inc -=1
                else:
                    flag_inc = False
                    flag_dec = True

            if flag_dec:
                if cnt_dec != 0:
                    set_lr(optimizer,0.5)
                    print(f'Learning rate decreased to {get_lr(optimizer)} : {cnt_dec}')
                    cnt_dec-=1
                else:
                    flag_dec = False

            if (cnt >= super_epochs and (not flag_inc) and (not flag_dec)):
                cnt_dec = 3
                cnt_inc = 3
                cnt=0
                # check if all losses stored in accumalated losses are close enough
                mean_loss = sum(accumulated_loss) / len(accumulated_loss)
                variance_loss = sum((x - mean_loss) ** 2 for x in accumulated_loss) / len(accumulated_loss)
                print("var loss: ",variance_loss)
                if variance_loss < self.threshold:
                    set_lr(optimizer,2)
                    print(f'Learning rate increased to {get_lr(optimizer)} : {cnt_inc}')
                    cnt_inc -=1
                    flag_inc = True
                accumulated_loss=[]
                


            model_local.eval()
            test_acc = 0
            test_loss = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model_local(data)
                    test_loss += self.criterion(output, target).item()

                    test_acc += 100*torch.sum(torch.argmax(output, dim=1) == target).item()/len(target)

            train_loss = train_loss/len(self.train_loader)
            test_loss = test_loss/len(self.test_loader)

            train_loss_optim.append(train_loss)
            train_acc_optim.append(train_acc/len(self.test_loader))

            test_loss_optim.append(test_loss)
            test_acc_optim.append(test_acc/len(self.test_loader))

            if (epoch%10==9):
                print(f'{epoch+1 :>6} {train_loss :>25} {test_loss :>25}')

        return train_loss_optim, test_loss_optim, train_acc_optim, test_acc_optim
    
    def train_all_optim(self, model):
        model.apply(initialize_weights_xavier)
        for opt in optimizers_dict:
            print(f"Optimizer : {opt}")
            # self.model = None
            # self.model = model
            # self.optimizer = optimizers_dict[optimizer](self.model.parameters(), self.lr)


            train_loss, test_loss, train_acc, test_acc = self.train(model, opt)

            self.train_loss_log[opt] = train_loss
            self.train_acc_log[opt] = train_acc

            self.test_loss_log[opt] = test_loss
            self.test_acc_log[opt] = test_acc

        log = {
            'train_losses' : self.train_loss_log,
            'train_accs' : self.train_acc_log,
            'test_losses' : self.test_loss_log,
            'test_accs' : self.test_acc_log,
        }

        return log

    def plot_loss_graphs(self):
        epoch_x = [i for i in range(1, self.epochs+1)]
        
        # plot loss curves for train and validation in two separate graphs
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        for key in self.train_loss_log:
            axs[0].plot(epoch_x, self.train_loss_log[key], label=key)
        axs[0].set_title('Train Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].grid()
        axs[0].legend()

        for key in self.test_loss_log:
            axs[1].plot(epoch_x, self.test_loss_log[key], label=key)
        axs[1].set_title('Test Loss')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].grid()
        axs[1].legend()

        return         
    
    def plot_acc_graphs(self):
        epoch_x = [i for i in range(1, self.epochs+1)]
        
        # plot loss curves for train and validation in two separate graphs
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        for key in self.train_acc_log:
            axs[0].plot(epoch_x, self.train_acc_log[key], label=key)
        axs[0].set_title('Train Accuracies')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Accuracy')
        axs[0].grid()
        axs[0].legend()

        for key in self.test_acc_log:
            axs[1].plot(epoch_x, self.test_acc_log[key], label=key)
        axs[1].set_title('Test Accuracies')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].grid()
        axs[1].legend()

        return    