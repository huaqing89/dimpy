import dim
from ..operate import Operate
from ..constant import Constant

class MSELossOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MSELossOperate,self).__init__(left,right,"mseLoss",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = dim.autograd.MulOperate.wrapper(self.left,prevOp)
    part2 = dim.autograd.MulOperate.wrapper(self.right,prevOp)
    part3 = self.left.partGrad(partial,part1)
    part4 = self.right.partGrad(partial,part2)
    rst = dim.autograd.AddOperate.wrapper(part3,part4)        
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "mseLoss("+self.left.expression()+")"
    self._expressionStr = rst
    return rst
   
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.mseLosss(self.left.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return MSELossOperate(left,right,args,name)

