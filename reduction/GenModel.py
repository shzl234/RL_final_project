
import torch

from myVAE import VanillaVAE

# https://zhuanlan.zhihu.com/p/577778277 diffusion!!!

class GEN_RED:

    def __init__(self,mode='VAE',name='random'):
        self.dir='reduction_weight/'+name+'_'+mode+'.pth'
        if mode=='VAE':
            self.model=VanillaVAE(64,10)
            # kld_weight = 0.00025
        
        elif mode=='NormFlow':  
            # to be done
            self.model=VanillaVAE(64,32)
        else:
            # to be done
            self.model=VanillaVAE(64,32)

    
    def red(self,x):


        return self.model.encoder(x)

    def red_noise(self,x):
        mu, log_var = self.model.encode(x)
        # assert False
        z = self.model.reparameterize(mu, log_var)
        # self.model.encode(x)
        return z
        # return self.model.encode(x)
    def rec(self,h):

        return self.model.decode(h)
    # def rec_by_noise()
    def load(self):
        self.model.load_state_dict(torch.load(self.dir))

        
    def save(self):
        torch.save(self.model.state_dict(),self.dir)
    
if __name__=='__main__':
    model=GEN_RED('VAE')
    x=torch.randn((123,64))
    h=model.red(x)
    model.load()

    h_noi=model.red_noise(x)
    print(h.shape)
    print(h_noi.shape)

    x_re=model.rec(h)
    # x_noi=model.rec_no

    print(x_re.shape)
