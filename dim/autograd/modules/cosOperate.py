import dim
from ..operate import Operate
from ..constant import Constant

class CosOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(CosOperate,self).__init__(left,right,"cos",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))

    part1 = dim.autograd.SinOperate.wrapper(self.left,None)
    part2 = dim.autograd.MulOperate.wrapper(Constant(-1),part1)
    part3 = dim.autograd.MulOperate.wrapper(part2,prevOp)
    rst = self.left.partGrad(partial,part3)
    self._grads[partial.name]=rst
    return rst
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "cos("+self.left.expression()+")"
    self._expressionStr = rst
    return rst

  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.cos(self.left.eval())
    self._data = rst
    return rst
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return CosOperate(left,right,args,name)

