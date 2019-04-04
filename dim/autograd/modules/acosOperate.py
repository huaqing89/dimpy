import dim
from ..operate import Operate
from ..constant import Constant

class AcosOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AcosOperate,self).__init__(left,right,"acos",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    part1 = dim.autograd.PowOperate.wrapper(self.left,2)
    part2 = dim.autograd.SubOperate.wrapper(Constant(1),part1)
    part3 = dim.autograd.PowOperate.wrapper(part1,0.5)
    part4 = dim.autograd.DivOperate.wrapper(Constant(-1),part3)
    part5 = dim.autograd.MulOperate.wrapper(part4,prevOp)
    rst = self.left.partGrad(partial,part5)
    self._grads[partial.name]=rst
    return rst
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "acos("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst= dim.acos(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return AcosOperate(left,right,args,name)

