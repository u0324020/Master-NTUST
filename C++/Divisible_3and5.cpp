#include<iostream>
using namespace std;
//�p�⥿��ƳQ3�㰣�Ȫ��`�M(1.��J�Q���Ƥ����� 2.��J�Q����) 
int main()
{
	int a;
	int k;
	int b=0;
	cin >> k;
	for(int j=0;j<k;j++){//��J�Q���Ƥ�����
		cout << "Input : ";
		cin>>a;
		for(int i=0;i<=a;i++){
			if(i%3==0){
				//cout << "add:" <<i<<endl;
				b = b+i;
			}}}

	cout << "Total : "<< b<<endl;
	system("pause");
	return 0;
}
