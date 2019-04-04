import dim
from ..operate import Operate
from ..constant import Constant

class MaxUnpool1dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MaxUnpool1dOperate,self).__init__(left,right,"maxUnpool1d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception("not implemented")
    
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "maxUnpool1d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
 
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.maxUnpool1d(
        self.left.eval(),self.args["indices"],self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return MaxUnpool1dOperate(left,right,args,name)
