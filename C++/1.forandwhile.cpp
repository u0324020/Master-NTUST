#include <iostream>
using namespace std;

int main(){
	// for loop 
	int a = 1, i = 1;
	for(i=1;i<=10;i++){
		a*=i;
	}
	// while loop
	int b = 1, j = 1;
	while(j<=10){
		a*=j;
		j++;
	}
	cout<<"for loop ans = "<<i<<endl;
	cout<<"while loop ans = "<<j;
}


