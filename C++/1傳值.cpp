#include <iostream>
using namespace std;

void swap(int,int);                           // �Ƶ{���ϥΫe�n���ŧi�A�ڭn�ǤJ��Ӿ�ơA�ǥX�@�Ӿ��

int main()		                 //main()��a,b,c �P max()��a,b,c �O���@�˪��F��!!
{
	int a=3,b=4;
	swap(a,b); 
	cout << "a=" << a 			 
	<< ",b=" << b << endl;
	return 0;
}

void swap(int a, int b) 
{
	int c = a;		
	a = b;		 	
	b = c;
}
