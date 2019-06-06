import dim
from ..operate import Operate
from ..constant import Constant

class TOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(TOperate,self).__init__(left,right,"T",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    if (self.left.name==partial.name):
      part1 = dim.autograd.TOperate.wrapper(prevOp,None)
      rst = part1
    else:
      part1 = dim.autograd.TOperate.wrapper(prevOp,None)
      part2 = self.left.partGrad(partial,part1)
      part3 = dim.autograd.TOperate.wrapper(part2,None)
      rst = part3
    
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "T("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
   
  def eval(self,useCatch=True):
    if (useCatch and self.catch and self._data is not None): return self._data

    rst=self.left.eval(useCatch)
    if (not self.isNumber(rst)): rst=rst.t()
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return TOperate(left,right,args,name)
