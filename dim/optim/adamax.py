import dim
from .optimizer import Optimizer
class Adamax(Optimizer):
  def __init__(self,params,**kwargs):
    super(Adamax,self).__init__(params)
    self.lr  = kwargs.get("lr",0.01)
    self.rho = kwargs.get("rho", (0.9,0.999))
    self.eps = kwargs.get("eps",1e-08)
    self.weight_decay = kwargs.get("weight_decay",0)
  def step(self):
    for x in self.params:
      if (x.requiresGrad):
        if x.grad is None:continue
        state=self.state[x.gradFn.name]

        if x.grad.shape!=x.shape:
          grad=x.grad.view().mean(0)
        else:
          grad=x.grad.view()
        
        if "m" in state:
          state["m"]=self.rho[0]*state["m"] + (1-self.rho[0])*grad
        else:
          state["m"]=(1-self.rho[0])*grad
        if "u" in state:
          state["u"]=(self.rho[1]*state["u"]+ grad.abs()).max(keepdims=True)
        else:
          state["u"]= grad.abs()
        
        if "step" in state:
          state["step"]+=1
        else:
          state["step"]=1
        
        mh=state["m"]/(1-self.rho[0]**state["step"])
        delta = self.lr/(state["u"].sqrt()+self.eps)*mh

        x-=delta
        