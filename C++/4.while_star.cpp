#include <iostream>
using namespace std;
//���T��
void RegularTriangle(int n){
	int s=0; 
	while(s<n){//����Ƽ� 
		s+=1;
		int k=(n-s);
		while(k>=1){//�L�h�֪ť� = �`���-�ثe��� 
			k-=1;
			cout << " "; }
			int j=0;
		while(j<s){//�C�Ʀh�֬P�P 
			j+=1;
			cout<<"*";}
		cout<<endl;}
		cout<<endl;//�L���@��Y���� 
}
//�ˤT�� 
void InvertedTriangle(int n){
	int i=n+1;
	while(i>0){ //����Ƽ� 
		i-=1;
		int j=0;
		while(j<i){ //����C�ƬP�P�� 
			j+=1;
			cout<<"*";
		}
		cout<<"\n"; //�L���@��Y���� 
	}
}
int main(){  
	int n;
	cin>>n;
	RegularTriangle(n);
	InvertedTriangle(n);
	system("pause");
	return 0;
}

