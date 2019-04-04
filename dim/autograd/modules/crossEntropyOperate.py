import dim
from ..operate import Operate
from ..constant import Constant

class CrossEntropyOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(CrossEntropyOperate,self).__init__(left,right,"crossEnropty",args,name)

  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = dim.autograd.CrossEntropyDeriOperate.wrapper(self.left,self.right)
    part2 = dim.autograd.MulOperate.wrapper(part1,prevOp)
    part3 = self.left.partGrad(partial,part2)
    rst = part3        
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "crossEntropy("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
   
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.nn.functional.crossEntropy(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return CrossEntropyOperate(left,right,args,name)

