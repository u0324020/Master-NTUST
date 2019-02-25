#encoding:UTF-8 <--中文編碼
#Url參數的Request
from flask import Flask, request 
app = Flask(__name__)

@app.route('/<name>/<age>')#定義Url參數名稱
def index(name,age):#把值傳入Function
	return ("My name is %s and %s old"%(name,age)) #return回網頁

	app.run(port=8080,debug=True)#http://127.0.0.1:8080/Jane/21