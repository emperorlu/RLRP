import  pandas  as pd
 
df=pd.read_excel('10工业.xls', sheet_name="表四")
data=df.head()#默认读取前5行的数据
print("获取到所有的值:\n{0}".format(data))#格式化输出