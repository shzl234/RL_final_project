import torch
import torch.nn as nn

class FineAdapter(nn.Module):
    def __init__(self, input_size, hidden_size=30):
        super(FineAdapter, self).__init__()
        # method like group convolution
        self.d1=int(input_size/3)
        self.d2=int(input_size*2/3)-self.d1
        self.d3=input_size-self.d1-self.d2
        self.adapter_fc1 = nn.Linear(self.d1, hidden_size)
        self.adapter_fc2 = nn.Linear(self.d2, hidden_size)
        self.adapter_fc3=nn.Linear(self.d3,hidden_size)

        self.adapter_if1=nn.Linear(hidden_size,self.d1)
        self.adapter_if2=nn.Linear(hidden_size,self.d2)
        self.adapter_if3=nn.Linear(hidden_size,self.d3)
        # self.relu = nn.ReLU()
        self.adapter_relu = nn.LeakyReLU()


    def forward(self, x):
        # x = self.fc1(x)
        # for i in range(3):
        t1=self.adapter_relu(self.adapter_if1(self.adapter_relu(self.adapter_fc1(x[:,0:self.d1]))))
        t2=self.adapter_relu(self.adapter_if2(self.adapter_relu(self.adapter_fc2(x[:,self.d1:self.d1+self.d2]))))
        t3=self.adapter_relu(self.adapter_if3(self.adapter_relu(self.adapter_fc3(x[:,self.d1+self.d2:]))))

        out=torch.cat([t1,t2,t3],dim=-1)
        return out
    

class CoarseAdapter(nn.Module):
    def __init__(self, input_size, hidden_size=10):
        super(CoarseAdapter, self).__init__()
        self.adapter_inlc = nn.Linear(input_size, hidden_size)
        self.adapter_fc1c=nn.Linear(hidden_size,hidden_size)
        self.adapter_fc2c = nn.Linear(hidden_size, hidden_size)
        self.adapter_fc3c=nn.Linear(hidden_size,hidden_size)
        self.adapter_oulc=nn.Linear(hidden_size,input_size)
        # self.relu = nn.ReLU()
        self.adapter_reluc = nn.LeakyReLU()
        

    def forward(self, x):
        t0=self.adapter_reluc(self.adapter_inlc(x))
        t1=self.adapter_reluc(self.adapter_fc1c(t0))
        t2=self.adapter_reluc(self.adapter_fc2c(t1))
        t3=self.adapter_reluc(self.adapter_fc3c(t2))
        out=self.adapter_reluc(self.adapter_oulc(t3))

            
        return out