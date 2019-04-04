import dim
from ..operate import Operate
from ..constant import Constant

class ReluOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(ReluOperate,self).__init__(left,right,"relu",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = dim.autograd.ReluDeriOperate.wrapper(self.left,None)
    part2 = dim.autograd.MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3
    self._grads[partial.name]=rst
    return rst  

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "relu("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.relu(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return ReluOperate(left,right,args,name)
