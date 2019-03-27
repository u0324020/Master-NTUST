#include <iostream>
using namespace std;

int GCD(int a,int b)
{
	int t;
	while (b !=0)
	{
		t = b;
		b = a%b ;
		a = t;
	}
	return (a);
}

int LCM (int a, int b )
{
	return (a*b / GCD(a,b));
}

int main()
{
	int a,b;
	cout<<"Please Enter Two Integer:\n";
	cin>>a>>b;
	cout<<"GCD: "<<GCD(a,b)<<endl;
	cout<<"LCM: "<<LCM(a,b)<<endl; 
}
