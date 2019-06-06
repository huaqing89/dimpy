import dim
from .optimizer import Optimizer
class RMSprop(Optimizer):
  def __init__(self,params,**kwargs):
    super(RMSprop,self).__init__(params)
    self.lr  = kwargs.get("lr",0.01)
    self.rho = kwargs.get("rho", 0.99)
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
        
        if "g2" in state:
          state["g2"]=self.rho*state["g2"] + (1-self.rho)*grad**2
        else:
          state["g2"]= (1-self.rho)*grad**2
        
        part1=state["g2"].sqrt()+self.eps
        x-=self.lr / part1 * grad