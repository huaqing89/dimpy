import dim
from ..operate import Operate
from ..constant import Constant

class AddOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(AddOperate,self).__init__(left,right,"add",args,name)

  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    part1= dim.autograd.AddOperate.wrapper(self.left.partGrad(partial,prevOp),self.right.partGrad(partial,prevOp))
    rst=part1
    self._grads[partial.name]=rst
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    rst = "("+self.left.expression() + "+" + self.right.expression()+")"
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.add(self.left.eval(),self.right.eval())
    self._data = rst
    return rst

  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if (left.type=="Constant" and (left.data==0).all()): return right
    if (right.type=="Constant" and (right.data==0).all()): return left
    if (left.type=="Constant" and right.type=="Constant"): return Constant(dim.add(left.data,right.data))
    return AddOperate(left,right,args,name)
