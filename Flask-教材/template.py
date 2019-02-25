#encoding:UTF-8 <--中文編碼
#templates
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def index(): #index function
	return render_template("index.html")#傳回templates底下的index.html檔畫面

@app.route("/login", methods=["GET"]) #定義URL GET
def login(): #login function

  if request.method == "GET": 
  	  return render_template("login.html") 

@app.route("/change", methods=["GET", "POST"]) #定義兩種URL方法
def change(): #change function

  if request.method == "GET":
  	  return render_template("change.html")

  if request.method == "POST": 
    name = request.form.get("name") #從html的form取name值
    return render_template("change.html", name=name)#將name值丟到form.html中

app.run(port=8080,debug=True)#http://127.0.0.1:8080