import xlrd  
 
workbook=xlrd.open_workbook("10工业.xls") 


'''对workbook对象进行操作'''
 
#获取所有sheet的名字
names=workbook.sheet_names()
print(names) #['各省市', '测试表']  输出所有的表名，以列表的形式
 
#通过sheet索引获得sheet对象
worksheet=workbook.sheet_by_index(0)
print(worksheet)  #<xlrd.sheet.Sheet object at 0x000001B98D99CFD0>
 
#通过sheet名获得sheet对象
worksheet=workbook.sheet_by_name("各省市")
print(worksheet) #<xlrd.sheet.Sheet object at 0x000001B98D99CFD0>
 
#由上可知，workbook.sheet_names() 返回一个list对象，可以对这个list对象进行操作
sheet0_name=workbook.sheet_names()[0]  #通过sheet索引获取sheet名称

