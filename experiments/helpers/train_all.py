import torch
import torch.optim as optim

import matplotlib.pyplot as plt

import os
os.chdir('/home/raid/Desktop/Shubh/DLProject/experiments/')

from optimizers.customAdam import customAdam


optimizers_dict = { "Adam_torch" : optim.Adam,
                    "Adam_custom" : customAdam }


# Helper class user for training and testing
class TrainerAll():
    def __init__(self, train_loader, test_loader, criterion, epochs=50, lr=0.001):
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = criterion
        self.epochs = epochs

        self.lr = lr
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loss_log = {}
        self.train_acc_log = {}

        self.test_loss_log = {}
        self.test_acc_log = {}

    def train(self):
        # defining local variables for the class
        train_loss_optim = []
        train_acc_optim = []
        test_loss_optim = []
        test_acc_optim = []

        # train loop
        self.model.train()

        for epoch in range(self.epochs):
            train_loss = 0

            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_acc = 100*torch.sum(torch.argmax(output, dim=1) == target).item()/len(target)

            self.model.eval()

            test_loss = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += self.criterion(output, target).item()

                    test_acc = 100*torch.sum(torch.argmax(output, dim=1) == target).item()/len(target)

            train_loss = train_loss/len(self.train_loader)
            test_loss = test_loss/len(self.test_loader)

            train_loss_optim.append(train_loss)
            train_acc_optim.append(train_acc)

            test_loss_optim.append(test_loss)
            test_acc_optim.append(test_acc)

            if (epoch%10==9):
                print(f'{epoch+1 :>6} {train_loss :>25} {test_loss :>25}')

        return train_loss_optim, test_loss_optim, train_acc_optim, test_acc_optim
    
    def train_all_optim(self, model):
        for optimizer in optimizers_dict:
            print(f"Optimizer : {optimizer}")

            self.model = model
            self.optimizer = optimizers_dict[optimizer](self.model.parameters(), self.lr)

            train_loss, test_loss, train_acc, test_acc = self.train()

            self.train_loss_log[optimizer] = train_loss
            self.train_acc_log[optimizer] = train_acc

            self.test_loss_log[optimizer] = test_loss
            self.test_acc_log[optimizer] = test_acc

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
        fig, axs = plt.subplots(1, 2, figsize=(16, 4))

        for key in self.train_loss_log:
            axs[0].plot(epoch_x, self.train_loss_log[key], label=key)
        axs[0].set_title('Train Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        for key in self.test_loss_log:
            axs[1].plot(epoch_x, self.test_loss_log[key], label=key)
        axs[1].set_title('Test Loss')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        return         