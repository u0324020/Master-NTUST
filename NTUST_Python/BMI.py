class BMI:
    def __init__(self,name,age,height,weight):
        self.__name = name
        self.__age = age
        self.__height = height
        self.__weight = weight
        
    def getBMI(self):
        return round(self.__weight/(self.__height/100)**2,2)

    def getStatus(self):
        BMI=round(self.__weight/(self.__height/100)**2,2)
        if BMI<18.5 :
            result = "過輕"
        elif BMI <=24 and BMI >=18.5:
            result = "正常"
        elif BMI <=27 and BMI >24 :
            result = "過重"
        elif BMI <=35 and BMI > 27:
            result = "肥胖"
        elif BMI > 35 :
            result = "極肥胖"
        return result

    def getTarget(self):
        BMI=round(self.__weight/(self.__height/100)**2,2)
        need = 0
        if BMI > 24 :
            need = round(self.__weight - (24 * ((self.__height/100)**2)),2)
            print("建議至少需減少",need,"公斤")
        elif BMI <18.5:
            need = round((18.5 * ((self.__height/100)**2)) - self.__weight,2)
            print("建議至少需增加",need,"公斤")
        return need

    def printReport(self):
        BMI=round(self.__weight/(self.__height/100)**2,2)
        print("====",self.__name,"體重狀況評估報告書=====")
        print("年齡=",self.__age,"身高:",self.__height,"公分","體重:",self.__weight,"公斤")
        print("評估結果:")
        print("BMI=",BMI,"體重狀況:",self.getStatus())
        self.getTarget()

again = True
while again :
    username = input("請輸入姓名:")
    userage = input("請輸入年齡:")
    userheight = eval(input("請輸入身高(公分):"))
    userweight=eval(input("請輸入體重(公斤):"))   
    user = BMI(username,userage,userheight,userweight)
    user.printReport()
    useragain = input("是否繼續輸入? 若要繼續輸入,請按'Y'或'y'")
    if useragain == "Y" or useragain == "y":
        continue
    else:
        break
