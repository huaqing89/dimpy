import dim
from ..operate import Operate
from ..constant import Constant

class SoftmaxOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SoftmaxOperate,self).__init__(left,right,"softmax",args,name)
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = dim.autograd.SoftmaxDeriOperate.wrapper(self,prevOp)
    part2 = self.left.partGrad(partial,part1)
    rst = part2        
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "softmax("+self.left.expression()+")"
    self._expressionStr = rst
    return rst

  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.softmax(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return SoftmaxOperate(left,right,args,name)

