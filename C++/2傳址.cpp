#include <iostream>
using namespace std;


int main()		                 //main()的a,b,c 與 max()的a,b,c 是不一樣的東西!!
{
	int a=3,b=4;
	int* p_a = &a;
	cout<<(p_a);
}


