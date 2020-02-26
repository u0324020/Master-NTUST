import os
# 這就是欲進行檔名更改的檔案路徑，路徑的斜線是為/，要留意下！
path = 'C:\\Users\\Jane\\Desktop\\NTU\\Scam\\Data\\Scam\\tss\\feedback\\tss_hit_20180905\\tss_detected_by_tss_hit_20180802\\test\\'
files = os.listdir(path)
print(files)  # 印出讀取到的檔名稱，用來確認自己是不是真的有讀到

k = 2600
for i in files:  # 因為資料夾裡面的檔案都要重新更換名稱
    oldname = path + i  # 指出檔案現在的路徑名稱，[n]表示第n個檔案
    # 在本案例中的命名規則為：年份+ - + 次序，最後一個.wav表示該檔案的型別
    k = k + 1
    newname = path + 'test\\' + str(k) + '.txt'
    os.rename(oldname, newname)
    print(oldname + '>>>' + newname)  # 印出原名與更名後的新名，可以進一步的確認每個檔案的新舊對應
