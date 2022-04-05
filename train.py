"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import copy
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from models import networks
from torch.utils.data import DataLoader

def evaluate(net,dataloader,opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lossDict = {}
    lossDict['D_A'] = []
    lossDict['G_A'] = []
    lossDict['cycle_A'] = []
    lossDict['idt_A'] = []
    lossDict['D_B'] = []
    lossDict['G_B'] = []
    lossDict['cycle_B'] = []
    lossDict['idt_B'] = []
    criterionGAN_ = networks.GANLoss('lsgan').to(device=device)
    criterionCycle_ = torch.nn.L1Loss().to(device=device)
    criterionIdt_ = torch.nn.L1Loss().to(device=device)   

    netG_A = net.netG_A
    netG_B = net.netG_B
    netD_A = net.netD_A
    netD_B = net.netD_B

    netG_A.eval()
    netG_B.eval()
    netD_A.eval()
    netD_B.eval()

    for batch in dataloader:
        real_A = batch['A'].to(device=device)
        real_B = batch['B'].to(device=device)
        fake_B = netG_A(real_A)
        rec_A = netG_B(fake_B)
        fake_A = netG_B(real_B)
        rec_B = netG_A(fake_A)

        # calculate validation scores for discriminators
        pred_real = netD_A(real_B)
        loss_D_real = criterionGAN_(pred_real,True)
        pred_fake = netD_A(fake_B.detach())
        loss_D_fake = criterionGAN_(pred_fake,False)
        lossDict['D_A'].append(((loss_D_real+loss_D_fake)*0.5).item())

        pred_real = netD_B(real_A)
        loss_D_real = criterionGAN_(pred_real,True)
        pred_fake = netD_B(fake_A.detach())
        loss_D_fake = criterionGAN_(pred_fake,False)
        lossDict['D_B'].append(((loss_D_real+loss_D_fake)*0.5).item())

        # calculate validation scores for generators
        lossDict['G_A'].append(criterionGAN_(netD_A(fake_B), True).item())
        lossDict['G_B'].append(criterionGAN_(netD_B(fake_A), True).item())   

        # calculate identity loss
        idt_A = netG_A(real_B)
        lossDict['idt_A'].append(criterionIdt_(idt_A, real_B).item() * opt.lambda_B * opt.lambda_identity)
        idt_B = netG_B(real_A)
        lossDict['idt_B'].append(criterionIdt_(idt_B, real_A).item() * opt.lambda_A * opt.lambda_identity)

        # calculate cycle loss
        lossDict['cycle_A'].append(criterionCycle_(rec_A,real_A).item() * opt.lambda_A)
        lossDict['cycle_B'].append(criterionCycle_(rec_B,real_B).item() * opt.lambda_B)

    for key in lossDict:
        lossDict[key] = np.mean(np.array(lossDict[key]))

    netG_A.train()
    netG_B.train()
    netD_A.train()
    netD_B.train()

    return lossDict

def writeCSV(csvPath, mode, trainingDict, validationDict):
    for key in trainingDict:
        newName = 'train_loss_'+key+'.csv'

        arrTraining = np.array(trainingDict[key])
        arrValidation = np.array(validationDict[key])

        lossMeanListTraining = np.cumsum(arrTraining)/np.cumsum(np.ones(len(arrTraining)))
        lossVarListTraining = [np.var(arrTraining[0:i]) for i in range(1,len(arrTraining)+1)]

        lossMeanListValidation = np.cumsum(arrValidation)/np.cumsum(np.ones(len(arrValidation)))
        lossVarListValidation = [np.var(arrValidation[0:i]) for i in range(1,len(arrValidation)+1)]   

        with open(csvPath+'/'+newName, mode=mode, newline='') as csvFile:
            csvWriter = csv.writer(csvFile)
            for i in range(len(lossMeanListTraining)):
                csvWriter.writerow([lossMeanListTraining[i], lossVarListTraining[i]])
                
        newName = 'validation_loss_'+key+'.csv'
        with open(csvPath+'/'+newName, mode=mode, newline='') as csvFile:
            csvWriter = csv.writer(csvFile)
            for i in range(len(lossMeanListValidation)):
                csvWriter.writerow([lossMeanListValidation[i], lossVarListValidation[i]])
    print("finished")



if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    valopt = copy.copy(opt)
    valopt.dataroot = opt.dataroot

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    validationDataset = create_dataset(valopt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    lossDict = {}
    lossDict['D_A'] = []
    lossDict['G_A'] = []
    lossDict['cycle_A'] = []
    lossDict['idt_A'] = []
    lossDict['D_B'] = []
    lossDict['G_B'] = []
    lossDict['cycle_B'] = []
    lossDict['idt_B'] = []

    valLossDict = {}
    valLossDict['D_A'] = []
    valLossDict['G_A'] = []
    valLossDict['cycle_A'] = []
    valLossDict['idt_A'] = []
    valLossDict['D_B'] = []
    valLossDict['G_B'] = []
    valLossDict['cycle_B'] = []
    valLossDict['idt_B'] = []
    valLossDict['xvalue'] = []
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing

            del data

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            losses = model.get_current_losses()
            lossDict['D_A'].append(losses['D_A'])
            lossDict['G_A'].append(losses['G_A'])
            lossDict['cycle_A'].append(losses['cycle_A'])
            lossDict['idt_A'].append(losses['idt_A'])
            lossDict['D_B'].append(losses['D_B'])
            lossDict['G_B'].append(losses['G_B'])
            lossDict['cycle_B'].append(losses['cycle_B'])
            lossDict['idt_B'].append(losses['idt_B'])
  
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            #if total_iters % 500 == 0:
            valLoss = evaluate(model,validationDataset,opt)
            valLossDict['xvalue'].append(total_iters)
            valLossDict['cycle_A'].append(valLoss['cycle_A'])
            valLossDict['cycle_B'].append(valLoss['cycle_B'])
            valLossDict['D_A'].append(valLoss['D_A'])
            valLossDict['D_B'].append(valLoss['D_B'])
            valLossDict['G_A'].append(valLoss['G_A'])
            valLossDict['G_B'].append(valLoss['G_B'])
            valLossDict['idt_A'].append(valLoss['idt_A'])
            valLossDict['idt_B'].append(valLoss['idt_B'])

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        ############################
        ############################
        save_dir = opt.checkpoints_dir+"/"+opt.name+'/plots'

        if not(os.path.isdir(save_dir)):
            os.mkdir(save_dir)
            mode = 'w'
        else:
            mode = 'w'

        # csv
        writeCSV(save_dir,mode,lossDict,valLossDict)


        # DISCRIMINATOR CURVES
        # training

        D_A = np.array(lossDict['D_A'])
        lossMeanListD_A = np.cumsum(D_A)/np.cumsum(np.ones(len(D_A)))
        lossVarListD_A = [np.var(D_A[0:i]) for i in range(1,len(D_A)+1)]
        D_B = np.array(lossDict['D_B'])
        lossMeanListD_B = np.cumsum(D_B)/np.cumsum(np.ones(len(D_B)))
        lossVarListD_B = [np.var(D_B[0:i]) for i in range(1,len(D_B)+1)]
        # validation
        val_D_A = np.array(valLossDict['D_A'])
        valLossMeanListD_A = np.cumsum(val_D_A)/np.cumsum(np.ones(len(val_D_A)))
        valLossVarListD_A = [np.var(val_D_A[0:i]) for i in range(1,len(val_D_A)+1)]
        val_D_B = np.array(valLossDict['D_B'])
        valLossMeanListD_B = np.cumsum(val_D_B)/np.cumsum(np.ones(len(val_D_B)))
        valLossVarListD_B = [np.var(val_D_B[0:i]) for i in range(1,len(val_D_B)+1)]
        #plot
        plt.plot(range(len(D_A)),lossMeanListD_A , '--', color="#191970",  label="training loss cbct discriminator")
        plt.fill_between(range(len(D_A)),lossMeanListD_A - lossVarListD_A, lossMeanListD_A + lossVarListD_A, color="#B0E0E6",alpha=0.5)
        plt.plot(range(len(D_B)),lossMeanListD_B , '--', color="#8B0000",  label="training loss ct discriminator B")
        plt.fill_between(range(len(D_B)),lossMeanListD_B - lossVarListD_B, lossMeanListD_B + lossVarListD_B, color="#FFA07A",alpha=0.5)

        plt.plot(valLossDict['xvalue'],valLossMeanListD_A , '--', color="#38761d",  label="validation loss cbct discriminator")
        plt.fill_between(valLossDict['xvalue'],valLossMeanListD_A - valLossVarListD_A, valLossMeanListD_A + valLossVarListD_A, color="#b6d7a8",alpha=0.5)
        plt.plot(valLossDict['xvalue'],valLossMeanListD_B , '--', color="#7f6000",  label="training loss ct discriminator B")
        plt.fill_between(valLossDict['xvalue'],valLossMeanListD_B - valLossVarListD_B, valLossMeanListD_B + valLossVarListD_B, color="#ffd966",alpha=0.5)

        plt.ylabel("E[log(D_y(y))]-E[log(1-D_y(G(x))]")
        plt.xlabel("training sample pairs")
        plt.legend()
        plt.savefig(save_dir+f'/Discriminator_{epoch}.png')

        plt.clf()
        plt.cla()
        plt.close()

        # GENERATOR CURVES
        # training
        D_A = np.array(lossDict['G_A'])
        lossMeanListD_A = np.cumsum(D_A)/np.cumsum(np.ones(len(D_A)))
        lossVarListD_A = [np.var(D_A[0:i]) for i in range(1,len(D_A)+1)]
        D_B = np.array(lossDict['G_B'])
        lossMeanListD_B = np.cumsum(D_B)/np.cumsum(np.ones(len(D_B)))
        lossVarListD_B = [np.var(D_B[0:i]) for i in range(1,len(D_B)+1)]
        # validation
        val_D_A = np.array(valLossDict['G_A'])
        valLossMeanListD_A = np.cumsum(val_D_A)/np.cumsum(np.ones(len(val_D_A)))
        valLlossVarListD_A = [np.var(val_D_A[0:i]) for i in range(1,len(val_D_A)+1)]
        val_D_B = np.array(valLossDict['G_B'])
        valLossMeanListD_B = np.cumsum(val_D_B)/np.cumsum(np.ones(len(val_D_B)))
        valLlossVarListD_B = [np.var(val_D_B[0:i]) for i in range(1,len(val_D_B)+1)]
        # plot
        plt.plot(range(len(D_A)),lossMeanListD_A , '--', color="#191970",  label="loss cbct generator")
        plt.fill_between(range(len(D_A)),lossMeanListD_A - lossVarListD_A, lossMeanListD_A + lossVarListD_A, color="#B0E0E6",alpha=0.5)
        plt.plot(range(len(D_B)),lossMeanListD_B , '--', color="#8B0000",  label="loss ct generator")
        plt.fill_between(range(len(D_B)),lossMeanListD_B - lossVarListD_B, lossMeanListD_B + lossVarListD_B, color="#FFA07A",alpha=0.5)


        plt.plot(valLossDict['xvalue'],valLossMeanListD_A , '--', color="#38761d",  label="validation loss cbct generator")
        plt.fill_between(valLossDict['xvalue'],valLossMeanListD_A - valLossVarListD_A, valLossMeanListD_A + valLossVarListD_A, color="#b6d7a8",alpha=0.5)
        plt.plot(valLossDict['xvalue'],valLossMeanListD_B , '--', color="#7f6000",  label="training loss ct generator")
        plt.fill_between(valLossDict['xvalue'],valLossMeanListD_B - valLossVarListD_B, valLossMeanListD_B + valLossVarListD_B, color="#ffd966",alpha=0.5)


        plt.ylabel("E[log(D_y(y))]-E[log(1-D_y(G(x))]")
        plt.xlabel("training sample pairs")
        plt.legend()
        plt.savefig(save_dir+f'/Generator_{epoch}.png')

        plt.clf()
        plt.cla()
        plt.close()

        # cycle curves
        D_A = np.array(lossDict['cycle_A'])
        lossMeanListD_A = np.cumsum(D_A)/np.cumsum(np.ones(len(D_A)))
        lossVarListD_A = [np.var(D_A[0:i]) for i in range(1,len(D_A)+1)]
        D_B = np.array(lossDict['cycle_B'])
        lossMeanListD_B = np.cumsum(D_B)/np.cumsum(np.ones(len(D_B)))
        lossVarListD_B = [np.var(D_B[0:i]) for i in range(1,len(D_B)+1)]
        # validation
        val_D_A = np.array(valLossDict['cycle_A'])
        valLossMeanListD_A = np.cumsum(val_D_A)/np.cumsum(np.ones(len(val_D_A)))
        valLlossVarListD_A = [np.var(val_D_A[0:i]) for i in range(1,len(val_D_A)+1)]
        val_D_B = np.array(valLossDict['cycle_A'])
        valLossMeanListD_B = np.cumsum(val_D_B)/np.cumsum(np.ones(len(val_D_B)))
        valLlossVarListD_B = [np.var(val_D_B[0:i]) for i in range(1,len(val_D_B)+1)]

        plt.plot(range(len(D_A)),lossMeanListD_A , '--', color="#191970",  label="cyclic loss person")
        plt.fill_between(range(len(D_A)),lossMeanListD_A - lossVarListD_A, lossMeanListD_A + lossVarListD_A, color="#B0E0E6",alpha=0.5)
        plt.plot(range(len(D_B)),lossMeanListD_B , '--', color="#8B0000",  label="cyclic generator waifu")
        plt.fill_between(range(len(D_B)),lossMeanListD_B - lossVarListD_B, lossMeanListD_B + lossVarListD_B, color="#FFA07A",alpha=0.5)

        plt.plot(valLossDict['xvalue'],valLossMeanListD_A , '--', color="#38761d",  label="validation cyclic loss person")
        plt.fill_between(valLossDict['xvalue'],valLossMeanListD_A - valLossVarListD_A, valLossMeanListD_A + valLossVarListD_A, color="#b6d7a8",alpha=0.5)
        plt.plot(valLossDict['xvalue'],valLossMeanListD_B , '--', color="#7f6000",  label="training cyclic loss waifu")
        plt.fill_between(valLossDict['xvalue'],valLossMeanListD_B - valLossVarListD_B, valLossMeanListD_B + valLossVarListD_B, color="#ffd966",alpha=0.5)

        plt.ylabel("E[||F(G(x))-x||]")
        plt.xlabel("training sample pairs")
        plt.legend()
        plt.savefig(save_dir+f'/Cycle_{epoch}.png')
        

        plt.clf()
        plt.cla()
        plt.close()

        # identity curves
        D_A = np.array(lossDict['idt_A'])
        lossMeanListD_A = np.cumsum(D_A)/np.cumsum(np.ones(len(D_A)))
        lossVarListD_A = [np.var(D_A[0:i]) for i in range(1,len(D_A)+1)]
        D_B = np.array(lossDict['idt_B'])
        lossMeanListD_B = np.cumsum(D_B)/np.cumsum(np.ones(len(D_B)))
        lossVarListD_B = [np.var(D_B[0:i]) for i in range(1,len(D_B)+1)]
        # validation
        val_D_A = np.array(valLossDict['idt_A'])
        valLossMeanListD_A = np.cumsum(val_D_A)/np.cumsum(np.ones(len(val_D_A)))
        valLlossVarListD_A = [np.var(val_D_A[0:i]) for i in range(1,len(val_D_A)+1)]
        val_D_B = np.array(valLossDict['idt_A'])
        valLossMeanListD_B = np.cumsum(val_D_B)/np.cumsum(np.ones(len(val_D_B)))
        valLlossVarListD_B = [np.var(val_D_B[0:i]) for i in range(1,len(val_D_B)+1)] 

        plt.plot(range(len(D_A)),lossMeanListD_A , '--', color="#191970",  label="identity loss cbct")
        plt.fill_between(range(len(D_A)),lossMeanListD_A - lossVarListD_A, lossMeanListD_A + lossVarListD_A, color="#B0E0E6",alpha=0.5)
        plt.plot(range(len(D_B)),lossMeanListD_B , '--', color="#8B0000",  label="identity loss ct")
        plt.fill_between(range(len(D_B)),lossMeanListD_B - lossVarListD_B, lossMeanListD_B + lossVarListD_B, color="#FFA07A",alpha=0.5)

        plt.plot(valLossDict['xvalue'],valLossMeanListD_A , '--', color="#38761d",  label="validation identity loss cbct")
        plt.fill_between(valLossDict['xvalue'],valLossMeanListD_A - valLossVarListD_A, valLossMeanListD_A + valLossVarListD_A, color="#b6d7a8",alpha=0.5)
        plt.plot(valLossDict['xvalue'],valLossMeanListD_B , '--', color="#7f6000",  label="training identity loss ct")
        plt.fill_between(valLossDict['xvalue'],valLossMeanListD_B - valLossVarListD_B, valLossMeanListD_B + valLossVarListD_B, color="#ffd966",alpha=0.5)

        plt.ylabel("E[||G(y)-y||]")
        plt.xlabel("training sample pairs")
        plt.legend()
        plt.savefig(save_dir+f'/Identity_{epoch}.png')

        plt.clf()
        plt.cla()
        plt.close()

        ############################
        ############################
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


