import torch

import matplotlib.pyplot as plt



# Helper class user for training and testing
class Trainer():
    def __init__(self, model, criterion, optimizer, lr=0.001):
        self.model = model

        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), lr=lr)

        self.criterion = criterion

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # defining local variables for the class
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []


    def train(self, train_loader, test_loader, epochs=50):
        self.epochs = epochs

        # train loop
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                accuracy = 100*torch.sum(torch.argmax(output, dim=1) == target).item()/len(target)

                # break

            self.train_loss.append(running_loss/len(train_loader))
            self.train_acc.append(accuracy)


            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += self.criterion(output, target).item()

                    val_acc = 100*torch.sum(torch.argmax(output, dim=1) == target).item()/len(target)

            self.val_loss.append(val_loss/len(test_loader))
            self.val_acc.append(val_acc)

            if (epoch%10==9):
                print(f'{epoch+1 :>6} {running_loss/len(train_loader) :>25} {val_loss/len(test_loader) :>25}')
                # print(f"Epoch: {epoch+1}/{epochs} ::     Train Loss: {running_loss/len(train_loader)}...     Val Loss: {val_loss/len(test_loader)}")                    

        return
        

    def plot_graph(self):
        epoch_x = [i for i in range(1, self.epochs+1)]
        
        # plot loss and accuracy curves for train and validation in two separate graphs
        fig, axs = plt.subplots(1, 2, figsize=(16, 4))

        axs[0].plot(epoch_x, self.train_loss, label='Train Loss')
        axs[0].plot(epoch_x, self.val_loss, label='Val Loss')
        axs[0].set_title('Loss')
        axs[0].legend()

        axs[1].plot(epoch_x, self.train_acc, label='Train Accuracy')
        axs[1].plot(epoch_x, self.val_acc, label='Val Accuracy')
        axs[1].set_title('Accuracy')
        axs[1].legend()

        return         