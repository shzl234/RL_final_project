import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from agent.adapter import CoarseAdapter,FineAdapter
import utils
from utils import cfg_env
# from 
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
from reduction.myVAE import VAE_RED 
from reduction.PCA import PCA_RED 
class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        
        super().__init__()
        Train_compress=True
        if Train_compress:
            self.compress=nn.Linear(17,11)
            self.compress2=nn.Linear(11,1024)
            self.relu=nn.LeakyReLU()
            self.compress3=nn.Linear(1024,12)
            self.comp=nn.Sequential(
            self.compress,
            self.compress2,
            self.relu,
            self.compress3
            
            )
            dire='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/reduction/vae_weight/pg_10000.pth'
            self.comp.load_state_dict(torch.load(dire),strict=False)
        if False:
            action_dim=3
        self.name=utils.cfg_env
        # assert False,2
        if True:
            obs_dim=11
        self.obs_pca=PCA_RED(11,'halfcheeth_obs')
        self.obs_pca.load_weight()
        if True:
            self.obs_vae=VAE_RED(17,11)
            self.obs_vae.load(self.obs_vae.vae,'/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/reduction/vae_weight/vae_half_obs11_90.pth')
            self.obs_vae.freeze(self.obs_vae.vae)
            self.obs_vae.vae=self.obs_vae.vae.cuda()
            self.obs_vae.vae.eval()
        
        if True:
            self.red_rand=nn.Linear(17,11)
            # nn.init.xavier_normal_(self.red_rand.bias)
            nn.init.xavier_normal_(self.red_rand.weight)

        self.dir='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/weight/'+self.name+'_actor.pth'
        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)
        # self.adapter=CoarseAdapter(hidden_dim,)
        self.adapter=FineAdapter(hidden_dim,)
        # self.adapter2=FineAdapter(hidden_dim,)
        # print(self.trunk)
        self.trunk=nn.Sequential(
            self.trunk[0:2],
            self.adapter,
            self.trunk[2:]
        )
        
        self.freezing=False
        self.loading=False
        # print("        ")
        # print(self.trunk)
        # assert False
        self.freezing=True
        if self.freezing:
            # assert False
            self.freeze(self.trunk)
            
        # self.save(self.trunk)
        # self.save(self.Q2)
        # self.loading=True
        # dd='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/reduction/vae_weight/sac_halfc_actor_800000.pth'
        # self.load(self,dd)
        # if self.loading:
            # self.load(self.Q1)
            # self.load(self.Q2)

        self.outputs = dict()
        # self.apply(utils.weight_init)
    
    def save_pg_compress(self,dir=None):
        if dir is None:
            torch.save(self.comp.state_dict(),'/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/reduction/vae_weight/pg_weight.pth')
        else :
             torch.save(self.comp.state_dict(),dir)
    def compresss(self,x):
        # out=self.compress(x)
        # out=self.compress2(out)
        # out=self.relu(out)
        # out=self.compress3(out)
        out=self.comp(x)
        return out

    def freeze(self,model):
        print(model.state_dict().keys())
    # print(model.)
        for name, param in model.named_parameters():
            
            print(name+':  ',param.requires_grad )
        print("--------------")
        for name, param in model.named_parameters():
            if '2.4' in name:
                continue
            if "adapter" not in name:
                param.requires_grad = False
            # if 
        
        for name, param in model.named_parameters():
            
            print(name+':  ',param.requires_grad )

    def load(self,model,dir=None):
        if dir is None:
            model.load_state_dict(torch.load(self.dir))
        else:
            model_state_dict=model.state_dict()
            state_dict=torch.load(dir)
            for key in state_dict.keys():
                print('key all',key)
                if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
                    print(key,'key!!!')
                    model_state_dict[key] = state_dict[key]
            # model.load_state_dict(torch.load(dir),strict=False)
            # assert False

        
    def save(self,model,dir=None):
        if dir==None:
            torch.save(model.state_dict(),self.dir)
        else:
            torch.save(model.state_dict(),dir)

    def forward(self, obs):
        # print(obs.dtype)
        # print(obs.shape)
        obs=obs[:,:11]+obs[:,1:]
        # assert False
        # print(type(obs[0,0]))
        if False:
            # print(self.obs_vae.vae)
            # print(next(self.obs_vae.vae.parameters()).device)
            # assert False
            # fg=obs.shape[0]==1
            #感觉这样会有大问题！！！
            # if fg:
                # obs=torch.cat([obs,obs],dim=0)
            # print(obs.shape)
            # with torch.no_grad():
            #     obs=self.red_rand(obs)
                # obs=self.obs_vae.vae.encoder(obs)+self.red_rand(obs)
                # obs=self.obs_vae.vae.encoder(obs)+obs[:,:11]+obs[:,6:]
                # obs=obs[:,:11]+obs[:,6:]
            # print(obs.shape)
            # print(obs)
            # obs=obs.cpu()
            # if fg:
            #     obs=obs[:1,:]
            # print(obs.shape)
            obs=obs.cpu()
            obs=self.obs_pca.pca.transform(obs)
            obs=torch.tensor(obs).to(dtype=torch.float32).to('cuda')
        # print(obs.dtype)
        # print(type(obs[0,0]),'22')
        if False:
            with torch.no_grad():
                # obs=self.red_rand(obs)
                obs=self.comp[0](obs)
                # assert False
        try:
            # mu, log_std=self.compresss(obs).chunk(2,dim=-1)
            mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        # try:
            # a=1
        except:
        # if True:
            print(self.trunk(obs))
            print(mu)
            print(obs)
            print(log_std)
        # assert False,'finish'
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                        1)

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)