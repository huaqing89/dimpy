import dim
from ..operate import Operate
from ..constant import Constant

class MulOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(MulOperate,self).__init__(left,right,"mul",args,name)
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    
    part1 = dim.autograd.MulOperate.wrapper(self.right,prevOp)
    part2 = self.left.partGrad(partial,part1)
    part3 = dim.autograd.MulOperate.wrapper(self.left,prevOp)
    part4 = self.right.partGrad(partial,part3)
    part5 = dim.autograd.AddOperate.wrapper(part2,part4)
    rst = part5
    self._grads[partial.name]=rst
    return rst
  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    part1=self.left.expression()
    part2=self.right.expression()
    if (part1=='-1'): rst= "-{}".format(part2)
    elif (part2=='-1'): rst= "-{}".format(part1)
    else: rst = "({}*{})".format(part1,part2)
    self._expressionStr = rst
    return rst
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.mul(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if (left.type=="Constant" and (left.data==0).all()): return Constant(0)
    if (right.type=="Constant" and (right.data==0).all()): return Constant(0)
    if (left.type=="Constant" and right.type=="Constant"): return Constant(dim.mul(left.data,right.data))
    if (left.type== "Constant" and (left.data==1).all()): return right
    if (right.type=="Constant" and (right.data==1).all()): return left
    
    return MulOperate(left,right,args,name)
