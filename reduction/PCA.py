import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.decomposition import PCA 
import joblib


class PCA_RED(PCA):
    def __init__(self,output_dim,name='random') -> None:
        # 
        self.pca=PCA(output_dim)
        self.output_dim = output_dim
        self.name=name
        self.dir='/home/zshccc/projectfile/DRL_hw/final_project/GA/pytorch_sac/weight/'+name+'_pca.pickle'
        
    def red_dem(self,x):
        return self.pca.transform(x)
    
    def rec_dem(self,x):
        return self.pca.inverse_transform(x)
    def train(self,X):
        X_train=self.pca.fit_transform(X)
        
        print(X.shape)
        print(X_train.shape)
        
        return X_train
    def save_weight(self):
        joblib.dump(self.pca, self.dir)
      
    def load_weight(self):
        print(self.dir)
        self.pca = joblib.load(self.dir)

    # def fit(self,X_data):
    #     N = len(X_data)
    #     H = torch.eye(n=N)-1/N*(torch.matmul(torch.ones(size=(N,1)),torch.ones(size=(1,N))))
    #     X_data = torch.matmul(H,X_data)
    #     _,_,v = torch.svd(X_data)
    #     self.base = v[:,:self.output_dim]

    # def fit_transform(self,X_data):
    #     self.fit(X_data)
    #     return self.transform(X_data)

    # def transform(self,X_data):
    #     return torch.matmul(X_data,self.base)

    # def inverse_transform(self,X_data):
    #     return torch.matmul(X_data,self.base.T)



if __name__=='__main__':
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train,dtype=torch.float)
    X_test = torch.tensor(X_test,dtype=torch.float)
    y_train = torch.tensor(y_train,dtype=torch.float)
    y_test = torch.tensor(y_test,dtype=torch.float)
    print(X_train.shape)  #torch.Size([1437, 64])
    print(X_test.shape) #torch.Size([360, 64])
    print(y_train.shape) #torch.Size([1437])
    print(y_test.shape)#torch.Size([360])


    pca = PCA_RED(20)
    X_train_pca = pca.train(X_train)
    pca.save_weight()

    pca = PCA_RED(20)
    pca.load_weight()
    print(X_train_pca.shape) #torch.Size([1437, 2])

    plt.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train)
    plt.figure()
    plt.subplot(331)

    for i,dim in enumerate([2,10,20,30,40,50,60]):
        # pca = PCA(dim)
        X_train_pca = pca.pca.transform(X_train)
        X_data = pca.pca.inverse_transform(X_train_pca)
        print(X_train_pca.shape)   #torch.Size([1437, 2])
        print(X_data.shape)  #torch.Size([1437, 64])
        break
        plt.subplot(2,4,i+1)
        # plt.imshow(X_data[0].view(8,8))
    plt.subplot(2,4,8)
    plt.imshow(X_train[0].view(8,8))
    # plt.show()
    pca = PCA_RED(20)
    # X_train_pca = pca.pca.fit_transform(X_train)
    pca.load_weight()
    model = GaussianNB()
    model.fit(X_train_pca,y_train)
    X_test_pca = pca.pca.transform(X_test)
    print(model.score(X_test_pca,y_test))
    
