#include<iostream>
#include<fstream>
using namespace std;
main(){
	fstream file;
	int price[3] = {100,90,80};
	char* id[3] = {"Apple","Banana","Cake"}; //�Y�n�sstring ������g�� �ŧi�r����а}�C���� 
											//�s�s��4bytes�O�����m 
	//printf("%s\n",id[0]);
	//printf("%p\n",id[0]);
	//printf("%s\n",0x00488000);
	file.open("menu.txt",ios::out);	//�g�J�Ҧ� 
	if(file.is_open()==false){
		cout << "Can't open\n";
		exit(1); //���_�{�� 
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


