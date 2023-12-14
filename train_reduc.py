import pickle
from replay_buffer import ReplayBuffer
import os
import torch
from reduction.PCA import PCA_RED 
from reduction.myVAE import VAE_RED
if __name__=='__main__':


    # dataset_path='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/dataset/sac_halfcheeth.pkl'
    dataset_path='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/dataset/sac_large_halfcheeth.pkl'
    with open(dataset_path, "rb") as f:
            datst = pickle.load(f)
    # print(datst[::5])
    # print(len(datst[::5]))
    # assert False
#     for i in range(len(datst)):
    obs=datst.obses[::50]
    
    acs=datst.actions[::50]
    # print(len(obs))
    # print(len(acs))
    # assert False
    Train_PCA=False
    Train_VAE=True
    if Train_VAE:
        # print(obs.shape)
        # action_vae=VAE_RED(6,)
        obs=torch.tensor(obs)
        obs_vae=VAE_RED(17,11,'half_obs_large11')
        obs_vae.train(obs,5000)

    if Train_PCA:
        action_pca=PCA_RED(3,'halfcheeth_acs_large')
        obs_pca=PCA_RED(11,'halfcheeth_obs')
        obs_pca.train(obs)
        action_pca.train(acs)
    #     obs_pca.load_weight()
    #     action_pca.load_weight()
    #     t1=obs_pca.pca.transform(obs[:10])
    #     t2=action_pca.pca.transform(acs[:10])
    #     print()
        obs_pca.save_weight()
        action_pca.save_weight()
    #     obss=obs_pca.pca.inverse_transform(t1)
    #     acss=action_pca.pca.inverse_transform(t2)
    #     print(obss.shape)
    #     print(type(obs))
    #     print(acss.shape)
