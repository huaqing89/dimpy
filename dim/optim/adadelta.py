import dim
from .optimizer import Optimizer
class Adadelta(Optimizer):
  def __init__(self,params,**kwargs):
    super(Adadelta,self).__init__(params)
    self.lr  = kwargs.get("lr",1)
    self.rho = kwargs.get("rho", 0.9)
    self.eps = kwargs.get("eps",1e-06)
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
          state["g2"]=dim.zeros(grad.shape)
        
        if not ("delta" in state):
          state["delta"]=(1-self.rho)*grad**2
        else:
          state["delta"]=self.rho*state["delta"] + (1-self.rho)* delta ** 2
         
        part1=state["delta"].sqrt()+self.eps
        part2=state["g2"].sqrt()+self.eps
        delta= part1/part2
               
        x-= delta
        