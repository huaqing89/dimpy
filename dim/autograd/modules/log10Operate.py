import math
import dim
from ..operate import Operate
from ..constant import Constant

class Log10Operate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(Log10Operate,self).__init__(left,right,"log10",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = dim.autograd.DivOperate.wrapper(Constant(1/math.log(10)),self.left)
    part2 = dim.autograd.MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "log10("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.log10(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return Log10Operate(left,right,args,name)
