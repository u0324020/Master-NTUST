#include<iostream>
#include<fstream> //�ɮ׳B�z���Y�� 
using namespace std;
//�����@��Ū�� 
main(){
	fstream file; //�إߤ@���ɮ׳B�z������file 
	char buffer[200]; //�Ψ��x�sŪ����r���e 
	file.open("Readme.txt",ios::in); //�}�Ҧ��W���ɮ� �Ω�Ū���Ҧ� 
	if(file.is_open()==false) //�p�G�ɮ׶}�ҥ��� 
		cout << "None\n";
	else{
		file.read(buffer,sizeof(buffer)); //Ū��������ƨé�J�}�C�� �]200�r 
		cout<<buffer<<"\n";
		file.close(); //�����ɮ� 
	}
	system("pause");
	return 0;
}


