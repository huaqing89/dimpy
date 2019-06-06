import sys
sys.path.append('..')

import dim

#x=dim.arange(12).reshape(3,4)
#y=dim.arange(3).reshape(3,1)
x=dim.rand(20,2)
y=(x[:,0].add(x[:,1])).reshape(20,1)
#x=x.normal(0)
#y=y.normal(0)
class Net(dim.nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.layer1=dim.nn.Sequential()
    self.layer1.addModule("fc1",dim.nn.Linear(2,10))
    self.layer1.addModule("relu1",dim.nn.ReLU())
    self.layer2=dim.nn.Sequential(
      dim.nn.Linear(10,5),
      dim.nn.ReLU()
    )
    self.layer1.addModule("layer2",self.layer2)
    #crossEntropyLoss
    self.layer1.addModule("out",dim.nn.Linear(5,10))
    #mseLoss
    self.layer1.addModule("out",dim.nn.Linear(10,1))
    #this.layer1.addModule("relu2",dim.nn.ReLU())
        
    self.addModule("all",self.layer1)

  def forward(self,x):
    return self.moduleList[0]["module"].forward(x)


net = Net()
print(net)
optim = dim.optim.Adam(net.parameters(),{"lr":0.00001})

preds=net.forward(x)
#criterion=dim.nn.CrossEntropyLoss()  
criterion=dim.nn.MSELoss()  
loss = criterion(preds,y)

for i in range(50000):
  loss.backward()
  optim.step()
  optim.zeroGrad()
  loss.gradFn.clearData()
  if (i+1)%5000==0:
    optim.lr*=0.1
    print("epoch=",i,"lr=",optim.lr,"loss=",loss.gradFn.eval().value())
    #print(list(x.sum() for x in net.parameters()))

'''
x1=dim.random.rand(20,2)
y1=(x1[:,0].add(x1[:,1])).reshape(20,1)
#x1=x1.normal(0)
#y1=y1.normal(0)
preds1=net(x1)
#preds1.print()

preds1=preds.gradFn.eval()
hat=preds1.argmax(1).reshape(y.shape)
total=y.shape[0]
correct=hat.eq(y1).sum()
accuracy=correct/total
print("准确度为:%{}".format(accuracy*100))
'''

'''
#conv2d layer
a=dim.arange(5*1*6*6).reshape(5,1,6,6)
conv2d=dim.nn.Sequential(
  dim.nn.Conv2d(1,3,3),
  dim.nn.MaxPool2d(2),
  dim.nn.ReLU()
)
m3=conv2d(a)
'''