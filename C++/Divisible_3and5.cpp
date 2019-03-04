#include<iostream>
using namespace std;
//計算正整數被3整除值的總和(1.輸入被除數之筆數 2.輸入被除數) 
int main()
{
	int a;
	int k;
	int b=0;
	cin >> k;
	for(int j=0;j<k;j++){//輸入被除數之筆數
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
