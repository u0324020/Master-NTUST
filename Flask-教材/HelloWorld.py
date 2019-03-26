#encoding:UTF-8 <--中文編碼
#Hello World 
from flask import Flask # import flask 的套件
app = Flask(__name__) # app是整個code的主角

@app.route('/') #根目錄 
def index(): # function name 

    return "Hello World!" #return值

app.run(port=8080,debug=True) #指定port number 及 更改code saving後會重run
#http://127.0.0.1:8080