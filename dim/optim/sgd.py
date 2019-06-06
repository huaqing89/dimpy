import dim
from .optimizer import Optimizer
class SGD(Optimizer):
  def __init__(self,params,**kwargs):
    super(SGD,self).__init__(params)
    self.lr  = kwargs.get("lr",0.001)
    self.weight_decay = kwargs.get("weight_decay",0)
    self.momentum = kwargs.get("moment",0.5)

  def step(self):
    for x in self.params:
      if (x.requiresGrad):
        if x.grad is None:continue
        state=self.state[x.gradFn.name]

        if x.grad.shape!=x.shape:
          grad=x.grad.view().mean(0)
        else:
          grad=x.grad.view()
        
        if self.weight_decay!=0:
          grad+=x.view()*self.weight_decay
          
        if "v" in state:
          state["v"]= self.momentum * state["v"] + grad
        else:
          state["v"]=grad
           
        x-=self.lr * state["v"]
        