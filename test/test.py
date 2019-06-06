import sys
sys.path.append("..")
import dim
import torch
import numpy as np
a=dim.random(0,1,10,10)
'''
a1=dim.random(0,1,10,20)
a2=dim.random(0,1,10,10)
a3=dim.randint(1,10,(10,1))
seqa=dim.nn.Sequential(dim.nn.Linear(10,1))
cta=dim.nn.MSELoss()
#a.setGrad()
a1.setGrad()
a2.setGrad()
b=torch.randn(10,10)
b1=torch.randn(10,20)
b2=torch.randn(10,10)
b3=torch.randint(1,10,(10,1))
seqb=torch.nn.Sequential(torch.nn.Linear(10,1))
ctb=torch.nn.MSELoss()
#b.requires_grad=True
b1.requires_grad=True
b2.requries_grad=True
c=np.random.randn(10,10)
c1=np.random.randn(10,20)
c2=np.random.randn(10,10)
'''
'''
%time a4=a.dot(a1).sum()
%time b4=b.mm(b1).sum()
%time c4=c.dot(c1).sum()

%time a4=(a+a2).mean()
%time b4=(b+b2).mean()
%time c4=(c+c2).mean()

print("[pred compare===>]")
%time preda=seqa(a)
%time predb=seqb(b)
print("[loss compare===>]")
%time lossa=cta(preda,a3)
%time lossb=ctb(predb,b3)
print("[backward compare===>]")
%time lossa.backward()
%time lossb.backward()

%timeit -n1000 preda=seqa(a);lossa=cta(preda,a3);lossa.backward()
%timeit -n1000 predb=seqb(b);lossb=ctb(predb,b3);lossb.backward()
'''