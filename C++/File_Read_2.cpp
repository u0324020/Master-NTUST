#include<iostream>
#include<fstream> //�ɮ׳B�z���Y�� 
using namespace std;
//�����r���v�@Ū�� 
main(){
	fstream file; //�إߤ@���ɮ׳B�z������file 
	char ch;
	file.open("Readme.txt",ios::in); //�}�Ҧ��W���ɮ� �Ω�Ū���Ҧ� 
	if(file.is_open()==false) //�p�G�ɮ׶}�ҥ��� 
		cout << "None\n";
	else{
		while(file.get(ch))//�r���v�@Ū�� 
			cout<<ch;
		cout<<endl;
		file.close(); //�����ɮ� 
	}
	system("pause");
	return 0;
}

