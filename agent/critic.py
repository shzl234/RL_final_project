import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from agent.adapter import CoarseAdapter,FineAdapter
import utils
# import hydra
from utils import cfg_env
#把 hidden_size调整为512，更快一点！！！
from reduction.PCA import PCA_RED 
from reduction.myVAE import VAE_RED 
class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        # print(action_dim)
        # assert False,action_dim
        # assert False,1
        if False:
            action_dim=3
        if True:
            obs_dim=11
        self.obs_pca=PCA_RED(11,'halfcheeth_obs')
        self.obs_pca.load_weight()
        if True:
            # self.obs_vae=VAE_RED(17,11)
            # self.obs_vae.load(self.obs_vae.vae,'/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/reduction/vae_weight/vae_half_obs11_90.pth')
            # self.obs_vae.freeze(self.obs_vae.vae)
            # self.obs_vae.vae=self.obs_vae.vae.cuda()
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
        #     # self.pg_red=
            # dire='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/reduction/vae_weight/pg_10000.pth'
            # self.comp.load_state_dict(torch.load(dire))
        # dd='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/reduction/vae_weight/sac_halfc_critic_800000.pth'
        # self.load(self,dd)

        if True:
            self.red_rand=nn.Linear(17,11)
            # nn.init.xavier_normal_(self.red_rand.bias)
            nn.init.xavier_normal_(self.red_rand.weight)



        self.name=utils.cfg_env
        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.dir='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/weight/'+self.name+'_critic.pth'
        self.freezing=False
        self.loading=False
        # print(self.Q1)
        # print(len(self.Q1))
        # self.adapter1=CoarseAdapter(hidden_dim,)
        # self.adapter2=CoarseAdapter(hidden_dim,)
        self.adapter1=FineAdapter(hidden_dim,)
        self.adapter2=FineAdapter(hidden_dim,)
        self.Q1=nn.Sequential(
            self.Q1[0:2],
            self.adapter1,
            self.Q1[2:]
        )
        self.Q2=nn.Sequential(
            self.Q2[0:2],
            self.adapter2,
            self.Q2[2:]
        )
        # print(self.Q1)
        # assert False
        if self.freezing:
            assert False
            self.freeze(self.Q1)
            self.freeze(self.Q2)
        # self.save(self.Q1)
        # self.save(self.Q2)
        # self.loading=True
        if self.loading:
            self.load(self.Q1)
            self.load(self.Q2)

        self.outputs = dict()
        # self.apply(utils.weight_init)

    def freeze(self,model):
        print(model.state_dict().keys())
    # print(model.)
        for name, param in model.named_parameters():
            
            print(name+':  ',param.requires_grad )
        for name, param in model.named_parameters():
            if "adapter" in name:
                param.requires_grad = False
        
        for name, param in model.named_parameters():
            
            print(name+':  ',param.requires_grad )

    def load(self,model,dir=None):
        if dir ==None:
            model.load_state_dict(torch.load(self.dir))
        else:
            model.load_state_dict(torch.load(dir))
        
    def save(self,model,dir=None):
        if dir==None:
            torch.save(model.state_dict(),self.dir)
        else:
            torch.save(model.state_dict(),dir)


    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        # obs=self.obs_pca.pca.transform(obs)
        # obs=torch.tensor(obs)
        obs=obs[:,:11]+obs[:,1:]


        if False:
            # with torch.no_grad():
                # obs=self.red_rand(obs)
                # obs=self.obs_vae.vae.encoder(obs)+self.red_rand(obs)
                # obs=self.obs_vae.vae.encoder(obs)+obs[:,:11]+obs[:,6:]
                # obs=obs[:,:11]+obs[:,6:]
            obs=obs.cpu()
            obs=self.obs_pca.pca.transform(obs)
            obs=torch.tensor(obs).to(dtype=torch.float32).to('cuda')
        if False:
            with torch.no_grad():
                # obs=self.red_rand(obs)

                # obs=self.red_rand(obs)
                obs=self.comp[0](obs)
                # assert False
        obs_action = torch.cat([obs, action], dim=-1)

       
        # 可以在这里通过切片来加入侧枝adapter
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        # print(obs_action.shape)
        # assert False,'!!!!!!'
        # print(q1.shape)
        # assert False
        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)
