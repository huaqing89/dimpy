from .optimizer import Optimizer
class Adam(Optimizer):
  def __init__(self,params,args={}):
    super(Adam,self).__init__()
    if (params):
      self.params = params
    self.lr  = args.get("lr",0.001)
    self.rho = args.get("rho", 0.9)
    self.eps = args.get("eps",1e-08)
    self.weight_decay = args.get("weight_decay", 0)
  def step(self):
    for x in self.params:
      if (x.requiresGrad):
        if x.grad.shape!=x.shape:
          x-=x.grad.mean(0)*self.lr
        else:
          x-=x.grad*self.lr
