import dim
from ..operate import Operate
from ..constant import Constant

class SubOperate(Operate):
  def __init__(self,left,right,args=None,name=None):
    super(SubOperate,self).__init__(left,right,"sub",args,name)
  
  def partGrad(self,partial,prevOp):
    if (partial.type!="Variable"): raise Exception("partial参数必须是Variable类型")
    if (self.catch and self._grads.get(partial.name,None)): return self._grads[partial.name]
    if (prevOp is None): prevOp=Constant(dim.ones(self.eval().shape))
    part1= dim.autograd.SubOperate.wrapper(self.left.partGrad(partial,prevOp),self.right.partGrad(partial,prevOp))
    rst=part1
    self._grads[partial.name]=rst
    return rst

  def expression(self):
    if (self.catch and self._expressionStr): return self._expressionStr
    part1= self.left.expression()
    part2= self.right.expression()
    if (part1=='0'):
      if (part2.slice(0,1)=='-'): rst= "{}".format(part2.slice(1))  
      else: rst= "(-{})".format(part2)
    elif (part2=='0'):
      rst= "{}".format(part1)
    else:
      rst = "({}-{})".format(part1,part2)

    self._expressionStr = rst
    return rst

  def eval(self):
    if (self.catch and self._data is not None): return self._data
    rst = dim.sub(self.left.eval(),self.right.eval())
    self._data = rst
    return rst
  
  @staticmethod
  def wrapper(left,right,args=None,name=None):
    if (right.type=="Constant" and (right.data==0).all()): return left
    if (left.type=="Constant" and right.type=="Constant"): return  Constant(dim.sub(left.data,right.data))
    if (left.isSame(right)): return Constant(1)
    
    return SubOperate(left,right,args,name)
