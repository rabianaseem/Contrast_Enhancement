import torch
import torch.optim as optim
from torchvision.utils import save_image
from datetime import datetime
import itertools
from libs.compute import *
from libs.constant import *
from libs.model import *
import torchvision.models as models
import gc
from libs.networks import *


clip_value = 1e8 
D_G_ratio = 50 # 50

if __name__ == "__main__":

    start_time = datetime.now()

    # Creating generator and discriminator
    generatorX = Generator()
    generatorX.load_state_dict(torch.load('./gan1_pretrain_100_4.pth', map_location=device))
    init_net(generatorX,'normal')

    generatorX_ = Generator_(generatorX)

    generatorX = nn.DataParallel(generatorX)

    generatorX_ = nn.DataParallel(generatorX_)
    generatorX.train()

    generatorY = Generator()
    init_net(generatorY,'normal')
    
    generatorY.load_state_dict(torch.load('./gan1_pretrain_100_4.pth', map_location=device))
    generatorY_ = Generator_(generatorY)
    
    generatorY = nn.DataParallel(generatorY)
   
    generatorY_ = nn.DataParallel(generatorY_)

    generatorY.train()    

    discriminatorY = Discriminator()
    init_net(discriminatorY,'normal')
    discriminatorY = nn.DataParallel(discriminatorY)

    discriminatorX = Discriminator()
    init_net(discriminatorX,'normal')
    discriminatorX = nn.DataParallel(discriminatorX)

    if torch.cuda.is_available():
        generatorX.cuda(device=device)
        generatorX_.cuda(device=device)
        generatorY.cuda(device=device)
        generatorY_.cuda(device=device)

        discriminatorY.cuda(device=device)
        discriminatorX.cuda(device=device)

    # Loading Training and Test Set Data
    trainLoader1, trainLoader2, trainLoader_cross, testLoader = data_loader()

   # MSE Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer_g = optim.Adam(itertools.chain(generatorX.parameters(), generatorY.parameters()), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_d = optim.Adam(itertools.chain(discriminatorY.parameters(),discriminatorX.parameters()), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    LambdaAdapt = LambdaAdapter(LAMBDA,D_G_ratio)

    batches_done = 0
    generator_loss = []
    discriminator_loss = []
    for epoch in range(NUM_EPOCHS_TRAIN):
        for i, (data, gt1) in enumerate(trainLoader_cross, 0):

            input, dummy = data
            groundTruth, dummy = gt1
            
            realInput = Variable(input.type(Tensor_gpu))   # stands for X
            realEnhanced = Variable(groundTruth.type(Tensor_gpu))   # stands for Y    
           
            fakeEnhanced = generatorX(realInput)   # stands for Y'
            fakeInput = generatorY(realEnhanced)           # stands for x'
                               
            if (LambdaAdapt.netD_change_times_1 > 0 and LambdaAdapt.netD_times >= 0 and LambdaAdapt.netD_times % LambdaAdapt.netD_change_times_1 == 0) or batches_done % 50 == 0: 
                LambdaAdapt.netD_times = 0

                recInput = generatorY_(torch.clamp(fakeEnhanced,0,1))     # stands for x''
                recEnhanced = generatorX_(torch.clamp(fakeInput,0,1))   # stands for y''

                set_requires_grad([discriminatorY,discriminatorX], False)

                # TRAIN GENERATOR
                generatorX.zero_grad()
                generatorY.zero_grad()

                ag = compute_g_adv_loss(discriminatorY,discriminatorX, fakeEnhanced,fakeInput)

                i_loss = computeIdentityMappingLoss(generatorX, generatorY, realEnhanced,realInput)

                c_loss = get_Perceptual_loss(realInput , recInput)

                g_loss = computeGeneratorLossFor2WayGan(ag, i_loss, c_loss)

                g_loss.backward()

                torch.nn.utils.clip_grad_value_(itertools.chain(generatorX.parameters(), generatorY.parameters()),clip_value)

                optimizer_g.step()
                
                del ag,i_loss,c_loss,recEnhanced,recInput #x2,y2 #,g_loss
                if torch.cuda.is_available() :   
                    torch.cuda.empty_cache()
                else:
                    gc.collect()       
          

            if batches_done % 500 == 0:
                # Training Network
                dataiter = iter(testLoader)
                gt_test, data_test = dataiter.next()
                input_test, dummy = data_test
                Testgt, dummy = gt_test
                
                realInput_test = Variable(input_test.type(Tensor_gpu))
                realEnhanced_test = Variable(Testgt.type(Tensor_gpu))
                with torch.no_grad():
                    psnrAvg = 0.0
                    for k in range(0, realInput_test.data.shape[0]):
                        save_image(realInput_test.data[k], "./models/train_images/Train_%d_%d_%d.png" % (epoch+1, batches_done+1, k+1),
                                    nrow=1,
                                    normalize=True)
                    torch.save(generatorX.state_dict(),
                                './models/train_checkpoint/gan_train_' + str(epoch) + '_' + str(i) + '.pth')
                    torch.save(discriminatorY.state_dict(),
                                './models/train_checkpoint/discriminator2_train_' + str(epoch) + '_' + str(i) + '.pth')
                    fakeEnhanced_test = generatorX(realInput_test)
                    test_loss = criterion( realEnhanced_test,fakeEnhanced_test  )
                     
                    psnr = 10 * torch.log10(1 / test_loss)
                    psnrAvg = psnr.mean()

                    print("Loss loss: %f" % test_loss)
                    print("PSNR Avg: %f" % (psnrAvg ))
                    f = open("./models/psnr_Score_trailing.txt", "a+")
                    f.write("PSNR Avg: %f" % (psnrAvg ))
                    f.close()

                    for k in range(0, fakeEnhanced_test.data.shape[0]):
                        save_image(fakeEnhanced_test.data[k],
                                    "./models/train_test_images/Train_Test_%d_%d_%d.png" % (epoch, batches_done, k),
                                    nrow=1, normalize=True)
                    
                del fakeEnhanced_test ,realEnhanced_test , realInput_test,  gt_test, data_test, dataiter,dummy ,Testgt, input_test
                
                if torch.cuda.is_available() :   
                    torch.cuda.empty_cache()
                else:
                    gc.collect()
            
            set_requires_grad([discriminatorY,discriminatorX], True)

            # TRAIN DISCRIMINATOR
            discriminatorX.zero_grad()
            discriminatorY.zero_grad()
                      
            #computing losses

            ad = compute_d_adv_loss(discriminatorY,realEnhanced,fakeEnhanced ) + compute_d_adv_loss(discriminatorX,realInput,fakeInput)

            gradient_penalty1 =  compute_gradient_penalty(discriminatorY, realEnhanced, fakeEnhanced) 
            gradient_penalty2 =  compute_gradient_penalty(discriminatorX, realInput,fakeInput)

            LambdaAdapt.update_penalty_weights(batches_done ,gradient_penalty1,gradient_penalty2)

            d_loss = computeDiscriminatorLossFor2WayGan(ad, LambdaAdapt.netD_gp_weight_1*gradient_penalty1 + LambdaAdapt.netD_gp_weight_2 * gradient_penalty2)
            
            d_loss.backward()

            torch.nn.utils.clip_grad_value_(itertools.chain(discriminatorY.parameters(),discriminatorX.parameters()),clip_value)

            optimizer_d.step()

            batches_done += 1
            LambdaAdapt.netD_times += 1
            print("Done training discriminator on iteration: %d" % i)

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ad loss: %f]  [gp1 loss: %f] [gp2 loss: %f][wp1 loss: %f] [wp2 loss: %f]" % (
                epoch + 1, NUM_EPOCHS_TRAIN, i + 1, len(trainLoader_cross), d_loss.item(), g_loss.item(),
                 ad,gradient_penalty1,gradient_penalty2,LambdaAdapt.netD_gp_weight_1,LambdaAdapt.netD_gp_weight_2 ))
            

            f = open("./models/log_Train.txt", "a+")
            f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n" % (
                epoch + 1, NUM_EPOCHS_TRAIN, i + 1, len(trainLoader_cross), d_loss.item(), g_loss.item()))
            f.close()
        

  # TEST NETWORK
    batches_done = 0
     # Training Network
    dataiter = iter(testLoader)
    #gt_test, data_test = dataiter.next()
    data_test, gt_test = dataiter.next()
    input_test, dummy = data_test

    Testgt, dummy = gt_test

    with torch.no_grad():
        psnrAvg = 0.0

        for j, (data, gt) in enumerate(testLoader, 0):
            input, dummy = data
            groundTruth, dummy = gt
            trainInput = Variable(input.type(Tensor_gpu))
            realImgs = Variable(groundTruth.type(Tensor_gpu))

            output = generatorX(trainInput)
            loss = criterion(output, realImgs)
            psnr = 10 * torch.log10(1 / loss)
            psnrAvg += psnr

            for k in range(0, output.data.shape[0]):
                save_image(output.data[k],
                           "./models/test_images/test_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1),
                           nrow=1,
                           normalize=True)
            for k in range(0, realImgs.data.shape[0]):
                save_image(realImgs.data[k],
                           "./models/gt_images/gt_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1),
                           nrow=1,
                           normalize=True)
            for k in range(0, trainInput.data.shape[0]):
                save_image(trainInput.data[k],
                           "./models/input_images/input_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1), nrow=1,
                           normalize=True)

            batches_done += 5
            print("Loss loss: %f" % loss)
            print("PSNR Avg: %f" % (psnrAvg / (j + 1)))
            f = open("./models/psnr_Score.txt", "a+")
            f.write("PSNR Avg: %f" % (psnrAvg / (j + 1)))
        f = open("./models/psnr_Score.txt", "a+")
        f.write("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))
        print("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))

    end_time = datetime.now()
    print(end_time - start_time)
