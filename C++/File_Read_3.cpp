#include<iostream>
#include<fstream> //�ɮ׳B�z���Y�� 
using namespace std;
//�C��Ū���@��
main(){
	fstream file; //�إߤ@���ɮ׳B�z������file 
	char buffer[80];
	file.open("Readme.txt",ios::in); //�}�Ҧ��W���ɮ� �Ω�Ū���Ҧ� 
	if(file.is_open()==false) //�p�G�ɮ׶}�ҥ��� 
		cout << "None\n";
	else{
		do{
			file.getline(buffer,sizeof(buffer));//�C��Ū���@���i�}�C 
			cout<<buffer<<endl;
		}while(!file.eof()); //�ɮ׵��� 
		file.close(); //�����ɮ� 
	}
	system("pause");
	return 0;
}


