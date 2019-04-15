#include<iostream>
#include<fstream> //檔案處理標頭檔 
using namespace std;
//全部一次讀取 
main(){
	fstream file; //建立一個檔案處理的物件file 
	char buffer[200]; //用來儲存讀取文字內容 
	file.open("Readme.txt",ios::in); //開啟此名稱檔案 用於讀取模式 
	if(file.is_open()==false) //如果檔案開啟失敗 
		cout << "None\n";
	else{
		file.read(buffer,sizeof(buffer)); //讀取全部資料並放入陣列中 設200字 
		cout<<buffer<<"\n";
		file.close(); //關閉檔案 
	}
	system("pause");
	return 0;
}


