class Optimizer(object):
  def __init__(self,params):
    if params is None:
      raise Exception("params args must be a iteror of Vector")
    self.params = params
    self.state={}
    for x in self.params:
      self.state[x.gradFn.name]={}

  def step(self):
    raise NotImplemented
  def zeroGrad(self):
    for x in self.params:
      if x.requiresGrad:
        x.gradClear()
