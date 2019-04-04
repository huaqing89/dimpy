import dim
from ..operate import Operate
from ..constant import Constant

class DivOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(DivOperate,self).__init__(left,right,"div",args,name)
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    part1 = dim.autograd.MulOperate.wrapper(self.right,prevOp)
    part2 = self.left.partGrad(partial,part1)
    part3 = dim.autograd.MulOperate.wrapper(self.left,prevOp)
    part4 = self.right.partGrad(partial,part3)
    part5 = dim.autograd.SubOperate.wrapper(part2,part4)
    part6 = dim.autograd.PowOperate.wrapper(self.right,Constant(2))
    part7 = dim.autograd.DivOperate.wrapper(part5,part6)
    rst = part7
    self._grads[partial.name]=rst
    return rst  

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    part1=self.left.expression()
    part2=self.right.expression()
    if (part2=='-1'): rst= "-{}".format(part1)
    else: rst = "({}/{})".format(part1,part2)
    self._expressionStr = rst
    return rst
  
  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.div(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if (left.type=="Constant" and (left.data==0).all()): return Constant(0)
    if (right.type=="Constant" and (right.data==0).all()): raise Exception("错误：除零的表达式") 
    if (left.type=="Constant" and right.type=="Constant"): return  Constant(dim.div(left.data,right.data))
    if (right.type=="Constant" and (right.data==1).all()): return left
    if (left.isSame(right)): return Constant(1)
    
    return DivOperate(left,right,args,name)
