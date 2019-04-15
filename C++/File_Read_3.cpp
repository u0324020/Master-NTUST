#include<iostream>
#include<fstream> //檔案處理標頭檔 
using namespace std;
//每次讀取一行
main(){
	fstream file; //建立一個檔案處理的物件file 
	char buffer[80];
	file.open("Readme.txt",ios::in); //開啟此名稱檔案 用於讀取模式 
	if(file.is_open()==false) //如果檔案開啟失敗 
		cout << "None\n";
	else{
		do{
			file.getline(buffer,sizeof(buffer));//每次讀取一行放進陣列 
			cout<<buffer<<endl;
		}while(!file.eof()); //檔案結尾 
		file.close(); //關閉檔案 
	}
	system("pause");
	return 0;
}


