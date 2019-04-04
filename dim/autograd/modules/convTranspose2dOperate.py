import dim
from ..operate import Operate
from ..constant import Constant

class ConvTranspose2dOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(ConvTranspose2dOperate,self).__init__(left,right,"convTranspose2d",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    raise Exception("not implemented")
    self._grads[partial.name]=rst
    return rst  
  
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "convTranspose2d("+self.left.expression()+","+self.right.expression()+")"
    self._expressionStr = rst
    return rst
   
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst=dim.nn.functional.convTranspose2d(self.left.eval(),self.right.eval(),1,self.args["padding"])
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    return ConvTranspose2dOperate(left,right,args,name)

