import sys
sys.path.append('..')

import autograd
import dim
dim=dim.Dim()

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
    self.layer1.addModule("layer2",this.layer2)
    #crossEntropyLoss
    this.layer1.addModule("out",dim.nn.Linear(1,10))
    #mseLoss
    this.layer1.addModule("out",dim.nn.Linear(5,1))
    #this.layer1.addModule("relu2",dim.nn.ReLU())
        
    self.addModule("all",this.layer1)

  def forward(self,x):
    return self.moduleList[0].module.forward(x)


net = dim.nn.Module1(Net)
preds=net.forward(x)
#let criterion=dim.nn.CrossEntropyLoss()  
criterion=dim.nn.MSELoss()  
optim = dim.optim.Adam(net.parameters(),{"lr":1})

for i in range(50000):
  loss = criterion.forward(preds,y)
  loss.backward()
  optim.step()
  optim.zeroGrad()
  loss.gradFn.setCatch(false)
  if (i%1000==0): print("epoch=",i,"loss=",loss.gradFn.eval().value)

y.print()
preds.gradFn.eval().print()


x1=dim.random.rand(20,2)
y1=(x1[:,0].add(x1[:,1])).reshape(20,1)
#x1=x1.normal(0)
#y1=y1.normal(0)
preds1=net.forward(x1)
#preds1.print()
y1.print()


preds1=preds.gradFn.eval()
hat=preds1.argmax(1).reshape(y.shape)
total=y.shape[0]
correct=hat.eq(y1).sum()
accuracy=correct/total
print("准确度为:%{}".format(accuracy*100))

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