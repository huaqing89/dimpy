import dim
from ..operate import Operate
from ..constant import Constant

class SumOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SumOperate,self).__init__(left,right,"sum",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = Constant(dim.ones(self.left.eval().shape))
    part2 = dim.autograd.MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "sum("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
    
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.sum(self.left.eval())
    self._data = rst
    return rst

  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return SumOperate(left,right,args,name)
