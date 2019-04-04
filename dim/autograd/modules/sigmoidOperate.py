import dim
from ..operate import Operate
from ..constant import Constant

class SigmoidOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SigmoidOperate,self).__init__(left,right,"sigmoid",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = dim.autograd.SigmoidDeriOperate.wrapper(self.left,None)
    part2 = dim.autograd.MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "sigmoid("+self.left.expression()+")"
    self._expressionStr = rst
    return rst 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.sigmoid(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return SigmoidOperate(left,right,args,name)
