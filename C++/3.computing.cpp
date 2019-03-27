#include <iostream>
using namespace std;

int add(int x,int y){
	return x+y;
}
int minus_N(int x,int y){
	return x-y;
}
int multiply(int x,int y){
	return x*y;
}
int divided_N(int x,int y){
	return x/y;
}

int main(){
	int x,y;
	cout<<"input 1:";
	cin>>x;
	cout<<"input 2:";
	cin>>y;
	if (x>=y){
		cout<<endl;
		cout<<"(+) = "<<add(x,y)<<endl;
		cout<<"(-) = "<<minus_N(x,y)<<endl;
		cout<<"(x) = "<<multiply(x,y)<<endl;
		cout<<"(/) = "<<divided_N(x,y)<<endl;
	}
	system("pause");
	return 0;
}


