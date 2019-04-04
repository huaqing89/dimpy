class Optimizer(object):
  def __init__(self,params=None):
    self.Optimizer(params)

  def Optimizer(self,params):
    if (params and (not isinstance(params,list))): params=[params]
    if (params):
      self.params = params
    return self
  def step(self):
    raise NotImplemented
  def zeroGrad(self):
    for x in self.params:
      if x.requiresGrad:
        x.gradClear()
  def Adam(self,params,args):
    if (params and (not isinstance(params,list))): params=[params]
    if (not params): params=self.params
    return Adam(params,args)    
