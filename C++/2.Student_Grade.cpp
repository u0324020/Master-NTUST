#include<iostream>
using namespace std;

int main()
{
    int n,i,a,b;
    cin >> n;
    i = 0;
    while( i < n )
    {	i = i+1;
        cin >> a ;
		cin >> b;
        cout <<"("<<i<<")"<< a+b << endl;
	}
    return 0;
}


