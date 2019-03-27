#include <iostream>
using namespace std;

void swap(int,int);                           // 副程式使用前要先宣告，我要傳入兩個整數，傳出一個整數

int main()		                 //main()的a,b,c 與 max()的a,b,c 是不一樣的東西!!
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
