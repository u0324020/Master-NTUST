#include<iostream>
#include<fstream> //檔案處理標頭檔 
using namespace std;
//全部字元逐一讀取 
main(){
	fstream file; //建立一個檔案處理的物件file 
	char ch;
	file.open("Readme.txt",ios::in); //開啟此名稱檔案 用於讀取模式 
	if(file.is_open()==false) //如果檔案開啟失敗 
		cout << "None\n";
	else{
		while(file.get(ch))//字元逐一讀取 
			cout<<ch;
		cout<<endl;
		file.close(); //關閉檔案 
	}
	system("pause");
	return 0;
}

