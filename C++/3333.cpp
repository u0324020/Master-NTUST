#include<iostream>
using namespace std;
int main()
{
	int a;
	int k;
	int b=0;
	cout << "Input : ";
	cin>>a;
	for( int i = 0; i<=a;i++){
	if(i%3==0){
		cout << "add:" <<i<<endl;
		b = b+i;
	}}
	cout << "Total : "<< b;
	system("pause");
	return 0;
}
