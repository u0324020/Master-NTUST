#include<iostream>
#include<fstream>
using namespace std;
//寫入
main(){
	fstream file;
	file.open("StudentID.txt",ios::out);	//寫入模式 
	if(file.is_open()==false)
		cout << "Can't open\n";
	else{
		file << "ID\t Name\n\n"; //寫入檔案 
		file << "03\t Jane\n";
		file << "06\t Eric\n";
		file << "12\t Danny\n";
		file << "13\t Brige\n";
		cout << "Done\n";
		file.close();
	}
	system("pause");
	return 0;
	}


