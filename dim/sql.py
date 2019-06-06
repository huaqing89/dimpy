class Dim:
  
    def toRCV(data):
      df=spark.createDataFrame(
        pd.DataFrame([(i,j,item) for i,eachitem in enumerate(data) for j,item in enumerate(eachitem)]),["r","c","v"])
      return df
      #dfa.join(dfb,dfa.r===dfb.r & dfa.c===dfb.c)
    def mul(dfa,dfb):
        dfa.createOrReplaceTempView('dfa')
        dfb.createOrReplaceTempView('dfb')
        dfc=spark.sql('''
           select dfa.r,dfa.c,(dfa.v*dfb.v) v
           from dfa join dfb on (dfa.r=dfb.r and dfa.c=dfb.c)
           order by dfa.r,dfa.c
        ''')
        return dfc
    def dot(dfa,dfb):
        dfa.createOrReplaceTempView('dfa')
        dfb.createOrReplaceTempView('dfb')
        dfc=spark.sql('''
            select dfa.r, dfb.c, sum(dfa.v*dfb.v) v
              from dfa join dfb on dfa.c=dfb.r
              group by dfa.r, dfb.c
              order by dfa.r,dfb.c
        ''')
        return dfc
    def T(df):
        return df.selectExpr("c as r","r as c",v)
    def _agg(df,axis=1,mode=None):
        if axis==None:
            return df.agg({"v":mode})
        if axis==0 :
            return df.groupby("c").agg({"v":mode})
        elif axis==1:
            return df.groupby("r").agg({"v":mode}) 
        else:
            raise Exception("error on axis")
    
    def std(df,axis=1):
        return _agg(df,axis,"std")
    def sum(df,axis=1):
        return _agg(df,axis,"sum")

#dot function
#print(a.dot(b1))
#time pd.DataFrame([(i,j,item) for i,eachitem in enumerate(a) \
#               for j,item in enumerate(eachitem)])
