import dim

class Module(object):
  def __init__(self):
    self.moduleList=[]
    self.eps=1e-5
  def addModule(self,name,module):
    if (module==None):
      module=name    
      name = str(self.count)
    self.moduleList.append({"name":name,"module":module})
    self.count=len(self.moduleList)
  
  def flatten(self,arr):
    for a in arr:
      if isinstance(a, list):
        yield from self.flatten(a)
      else:
        yield a
        
  def __str__(self):
    return "Modules(\n"\
           +"\n".join(list("  ({}):{}".format(x["name"],x["module"]) for x in self.moduleList))\
           +"\n)"        
  
  def _modules(self):
    for x in self.moduleList:
      if (len(x["module"].moduleList)!=0 and x["module"]!=self):
        yield from x["module"].modules()
      else:  
        yield x["module"] 
  def modules(self):
     mods=self.flatten(self._modules())
     return list(filter(lambda x:x is not None,mods))
  def _parameters(self):
    for x in self.moduleList:
      if (len(x["module"].moduleList)!=0 and x["module"]!=self):
        yield from x["module"].parameters()
      else:  
        yield getattr(x["module"],"params",None)  
  def parameters(self):
     params=self.flatten(self._parameters())
     return list(filter(lambda x:x is not None,params))

  def setParameters(self):
    pass

  def forward(self):
    print("must implement this function")
  
  def __call__(self,*args):
    return self.forward(*args)