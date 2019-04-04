import sys
sys.path.append("..")
import dim
x=dim.rand(50,5)
y=dim.randint(0,9,(50,1))
seq=dim.nn.Sequential(\
  dim.nn.Linear(5,4),
  dim.nn.ReLU(),
  dim.nn.Linear(4,1),
  dim.nn.ReLU()
)
pred=seq.forward(x)
#g=dim.nn.functional.crossEntropy(m2,y)
optim = dim.optim.Adam(seq.parameters(),{"lr":0.5})
criterion = dim.nn.MSELoss()
g=criterion.forward(pred,y)
for i in range(500):
  if (i):
    g.backward()
    optim.step()
    optim.zeroGrad()
  g.gradFn.setCatch(False)
  if (i%100==0):
    print("epoch=",i,"loss=",g.gradFn.eval().value())
    print(g.gradFn.expression())
