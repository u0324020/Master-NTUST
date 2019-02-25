#encoding:UTF-8 <--中文編碼
#Request and merge
from flask import Flask, request
app = Flask(__name__)
#方法一
@app.route('/<name>/<age>')
def index(name,age):
	return ("My name is %s and %s old"%(name,age))#http://127.0.0.1:8080/Jane/21
#方法二
@app.route('/Info/<ID>/<Name>/<Class>')#Info目錄
def Info(ID,Name,Class):#取URL網址再分析
	ID = request.values.get('ID')
	#Name = request.values.get('Name')
	#Class = request.values.get('Class')
	return ("ID = %s , Name = %s , Class = %s"%(ID,Name,Class))
	
app.run(port=8080,debug=True)#http://127.0.0.1:8080/Info/?ID=0324020&Name=Jane&Class=3B