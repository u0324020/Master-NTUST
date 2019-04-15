#include<iostream>
#include<fstream>
using namespace std;
main(){
	fstream file;
	int price[3] = {100,90,80};
	char* id[3] = {"Apple","Banana","Cake"}; //若要存string 必須改寫成 宣告字串指標陣列型式 
											//存連續4bytes記憶體位置 
	//printf("%s\n",id[0]);
	//printf("%p\n",id[0]);
	//printf("%s\n",0x00488000);
	file.open("menu.txt",ios::out);	//寫入模式 
	if(file.is_open()==false){
		cout << "Can't open\n";
		exit(1); //中斷程式 
	}
	else{
		file<<"item"<<"\t "<<"price\n\n";
		for(int i=0;i<3;i++){
			file<<id[i]<<"\t$"<<price[i]<<endl;
		}
		cout << "Done\n";
		file.close();
	}
	system("pause");
	return 0;
}


