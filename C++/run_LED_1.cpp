#include <iostream>
using namespace std;
//�Ʀr�]���O 
int main(){ 
	int jc = 0;
	for(int i=0;i<=30;i++){ //�����`�ƶq 
		cout << "\r"; //�C����^�̫e��A�L 
		for(int k=(30-i);k>=1;k--){//�ݥ��L�h�֪ť� 
			cout << " "; }
		for(int j=1;j<=i;j++){//�L�X�Ӧ�� 
			cout << j%10;}
		jc=1;
		while(1){//�ɶ����� 
			if(jc>50000000) break;
			jc++;}}
		cout<<endl;
		system("pause");
		return 0;
}


