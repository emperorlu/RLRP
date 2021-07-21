import xlrd  
 
workbook=xlrd.open_workbook("10工业.xls") 


 
#获取所有sheet的名字
names=workbook.sheet_names()
print(names) 
 
#通过sheet索引获得sheet对象,这里获取表四
worksheet=workbook.sheet_by_index(4)
print(worksheet)
 
# #通过sheet名获得sheet对象
# worksheet=workbook.sheet_by_name("各省市")
# print(worksheet)
 
# #由上可知，workbook.sheet_names() 返回一个list对象，可以对这个list对象进行操作
# sheet0_name=workbook.sheet_names()[0]  #通过sheet索引获取sheet名称


'''对sheet对象进行操作'''
name=worksheet.name  #获取表的姓名
print(name) #各省市
 
nrows=worksheet.nrows  #获取该表总行数
print(nrows)  #32
 
ncols=worksheet.ncols  #获取该表总列数
print(ncols) #13
