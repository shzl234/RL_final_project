import torch
# from models import BaseVAE
from reduction.base_vae import BaseVAE
from torch import nn
from torch.nn import functional as F
# from .types_ import *
from  torch import Tensor
from typing import List
from torch.utils.tensorboard import SummaryWriter


class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        buf_in_channels=in_channels
        self.latent_dim = latent_dim
        # assert False
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, latent_dim]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    # nn.Conv2d(in_channels, out_channels=h_dim,
                    #           kernel_size= 3, stride= 2, padding  = 1),
                    # 
                    nn.Linear(in_channels,h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            # print(hidden_dims[-(i+1)],hidden_dims[-(i + 2)])
            print(hidden_dims[i],hidden_dims[i + 1])
            modules.append(
                nn.Sequential(
                    # nn.ConvTranspose2d(hidden_dims[i],
                    #                    hidden_dims[i + 1],
                    #                    kernel_size=3,
                    #                    stride = 2,
                    #                    padding=1,
                    #                    output_padding=1),

                    nn.Linear(hidden_dims[i],hidden_dims[i + 1]),
                    # nn.Linear(hidden_dims[i],hidden_dims[i + 1]),
                    # nn.Linear()
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(hidden_dims[-1],
        #                                        hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(hidden_dims[-1], out_channels= 3,
        #                               kernel_size= 3, padding= 1),
        #                     nn.Tanh())
        self.final_layer=nn.Linear(hidden_dims[-1],buf_in_channels)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # print(input.shape)
        result = self.encoder(input)
        # print(result.shape)
        # result = torch.flatten(result, start_dim=1)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # result = result.view(-1, 512, 2, 2)
        # assert False
        
        result = self.decoder(result)
        # print(result.shape)
        # assert False
        result = self.final_layer(result)
        # assert False
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # print(input.shape)
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,#recons,input,mu,log_var) -> dict:
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # print(len(args))
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        # print(recons.shape)
        # print(input.shape)
        recons_loss =F.mse_loss(recons, input)

        kld_weight = 0.00025
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
class VAE_RED:
    def __init__(self,in_channels,latent_dim,name='random'):
        learning_rate=0.005

        self.vae=VanillaVAE(in_channels,latent_dim)
        self.opt=torch.optim.Adam(
            self.vae.parameters(),
            lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.dir='reduction/vae_weight/'+'vae_'+name+'.pth'

        
        log_dir = "/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/reduction/vae_train"
        self.tb = SummaryWriter(log_dir)

    def load(self,model,dir=None):
        if dir is None:

            model.load_state_dict(torch.load(self.dir))
        else:
            model.load_state_dict(torch.load(dir))
        

    def freeze(self,model):
        # for name, param in model.named_parameters():
            
        #     print(name+':  ',param.requires_grad )
        for name, param in model.named_parameters():
            
            param.requires_grad = False
        
        # for name, param in model.named_parameters():
            
        #     print(name+':  ',param.requires_grad ) 
    def save(self,model,dir=None):
        if dir is None:
            torch.save(model.state_dict(),self.dir)
        else:
            torch.save(model.state_dict(),dir)
    def train(self,dataset,epoch):
        bz=64

        mydataset=torch.utils.data.DataLoader(
            dataset,
            batch_size=bz,
            num_workers=2,
            shuffle=True,
            pin_memory=True,
        )

        for step in range(epoch):
            print(step,'!!!')
            print_tot=0
            print_rec=0
            print_kld=0
            for i,batch in enumerate(mydataset):
                if i%10000==0:
                    print('data now',i)
                self.opt.zero_grad()

                losstot=self.vae.loss_function(*self.vae(batch))
                
                
                loss=losstot['loss']
                rec_loss=losstot['Reconstruction_Loss']
                kld_loss=losstot['KLD']
                

                loss.backward()
                self.opt.step()
                print_kld+=kld_loss
                print_rec+=rec_loss
                print_tot+=loss.detach()
            
            # print(print_tot)
            # assert False
            self.tb.add_scalar("train_loss", print_tot.item(), global_step=step)
            self.tb.add_scalar("rec_loss", print_rec.item(), global_step=step)
            self.tb.add_scalar("kld_loss", print_kld.item(), global_step=step)
            if step%10==0:
                # print(step)
                name='half_obs11_'+str(step)
                
                dir='reduction/vae_weight/'+'vae_'+name+'.pth'
                print("Save to"+dir)
                self.save(self.vae,dir)
        




    
