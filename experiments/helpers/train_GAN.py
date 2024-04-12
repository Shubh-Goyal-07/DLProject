import torch
import torch.optim as optim
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import copy

import os
os.chdir('/home/raid/Desktop/Shubh/DLProject/experiments/')


optimizers_dict = {
                    # "Adam_custom" : customAdam5,
                    "Adam_torch" : optim.Adam,
                    # "RMS_torch" : optim.RMSprop,
                    "AdaGrad_torch" : optim.Adagrad
                }

real_label = 1.
fake_label = 0.
fixed_noise = torch.randn(128, 128, 1, 1)

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
class TrainerGAN():
    def __init__(self, train_loader, test_loader, criterion, epochs=10, lr=0.0002, threshold=4):

        train_loader = itertools.islice(train_loader, 780)
        train_loader_list = list(train_loader)
        
        test_loader = itertools.islice(test_loader, 75)
        test_loader_list = list(test_loader)

        self.train_loader = train_loader_list
        self.test_loader = test_loader_list

        self.threshold = threshold

        self.criterion = criterion
        self.epochs = epochs

        self.lr = lr
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loss_gen = {}
        self.train_loss_disc = {}


    def train(self, disc, gen, opt):

        # defining local variables for the class

        super_epochs = 3  #hyper-parameter
        cnt = 0
        cnt_inc = 0
        cnt_dec = 0
        flag_inc = False
        flag_dec = False
        accumulated_loss = []

        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.iters = 0

        # model_local = copy.deepcopy(model_local)
        disc_local = copy.deepcopy(disc)
        gen_local = copy.deepcopy(gen)

        optimizerD = optimizers_dict[opt](disc_local.parameters(), self.lr)
        optimizerG = optimizers_dict[opt](gen_local.parameters(), self.lr)


        for epoch in range(self.epochs):
            train_loss_disc = 0
            train_loss_gen = 0

            if (not flag_inc) and (not flag_dec):
                cnt += 1
            
            disc_local.train()
            gen_local.train()

            for i, data in enumerate(self.train_loader, 0):

                data = data[0].to(torch.float32).to(self.device)   # check if this is correct
                data_size = data.size(0)
                label = torch.full((data_size,), real_label, dtype=torch.float).to(self.device)
                optimizerD.zero_grad()
                
                # print("input shape: ",data.shape)
                
                outputD = disc_local(data).view(-1)

                # print("output shape: ",outputD.shape)

                errD_real = self.criterion(outputD,label)
                errD_real.backward()
                D_x = outputD.mean().item()

                ## Train with all-fake batch

                noise = torch.randn(data_size, 128, 1, 1).to(self.device)
                fake = gen_local(noise)
                label.fill_(fake_label).to(self.device)

                output = disc_local(fake.detach()).view(-1)

                errD_fake = self.criterion(output, label)

                errD_fake.backward()
                D_G_z1 = output.mean().item()

                errD = errD_real + errD_fake
                errD = errD

                optimizerD.step()

                gen_local.zero_grad()
                label.fill_(real_label).to(self.device)

                output2 = disc_local(fake).view(-1)

                errG = self.criterion(output2, label)
                errG = errG

                errG.backward()
                D_G_z2 = output2.mean().item()

                optimizerG.step()

                if i >=780 :
                    print(i)
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, self.epochs, i, len(self.train_loader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                train_loss_disc += errD.item()
                train_loss_gen += errG.item()
            self.G_losses.append(train_loss_gen)
            self.D_losses.append(train_loss_disc)

            accumulated_loss.append(train_loss_disc)

            if (flag_inc):
                if (cnt_inc!=0):
                    set_lr(optimizerD,2)
                    print(f'Learning rate increased to {get_lr(optimizerD)} : {cnt_inc}')
                    cnt_inc -=1
                else:
                    flag_inc = False
                    flag_dec = True

            if flag_dec:
                if cnt_dec != 0:
                    set_lr(optimizerD,0.5)
                    print(f'Learning rate decreased to {get_lr(optimizerD)} : {cnt_dec}')
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
                    set_lr(optimizerD,2)
                    print(f'Learning rate increased to {get_lr(optimizerD)} : {cnt_inc}')
                    cnt_inc -=1
                    flag_inc = True
                accumulated_loss=[]

        return self.G_losses , self.D_losses , copy.deepcopy(gen_local)
    
    def train_all_optim(self, disc, gen):
        gen.apply(initialize_weights_xavier)
        disc.apply(initialize_weights_xavier)

        # self.Gen = gen
        # self.Disc = disc
        model_dict = {}

        for opt in optimizers_dict:
            print(f"Optimizer : {opt}")

            gen_loss, disc_loss, trained_gen = self.train(disc,gen, opt)

            self.train_loss_gen[opt] = gen_loss
            self.train_loss_disc[opt] = disc_loss

            model_dict[opt] = {'generator': trained_gen}

        log = {
            'train_gen_losses' : self.train_loss_gen,
            'train_disc_losses' : self.train_loss_disc,
        }


        return log , model_dict

    def calc_fid_score(self, loader, inception_model, trained_gen):
        # print("HI")
        Gen = copy.deepcopy(trained_gen)

        inception_model = inception_model.to(self.device)

        real_features = []
        fake_features = []

        # Generate feature maps for whole dataset
        for real, target in loader:

            real = real.to(self.device)
            real_inc = self.resize_image(real)
            real_features.append(inception_model(real_inc).detach().cpu().numpy())

            # print("test_shape: ",real.shape)
            noise = torch.randn(128, 128, 1, 1).to(self.device)
            fake = Gen(noise)
            fake = self.resize_image(fake)
            fake_features.append(inception_model(fake).detach().cpu().numpy())

        real_features = np.concatenate(real_features)
        
        fake_features = np.concatenate(fake_features)

        fid = self.fid_score(real_features, fake_features)

        return fid
    
    def fid_score(self, real_features, fake_features):
        # FID Score calculations
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
        diff = mu_real - mu_fake
        cov_sqrt = sqrtm(sigma_real.dot(sigma_fake))
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
        fid_score = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
        return fid_score
    
    def resize_image(self, images):
        iter_size = images.size(0)
        # Resizing the images to 96x96 for inception model
        resize_transform = transforms.Resize((96, 96))
        images_resized = torch.stack([resize_transform(images[i]) for i in range(iter_size)])

        return images_resized

    def plot_losses(self,log):
        epoch_x = [i for i in range(1, self.epochs+1)]

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        mod1,mod2 = log
        for key in log[mod1]:
            axs[0].plot(epoch_x, log[mod1][key], label=key)
        axs[0].set_title('Generator Losses')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].grid()
        axs[0].legend()

        for key in log[mod2]:
            axs[0].plot(epoch_x, log[mod2][key], label=key)
        axs[0].set_title('Discriminator Losses')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].grid()
        axs[0].legend()

        return