#include <iostream>
#include <bitset>
//十進制轉二進制(如果輸入0即停止) 
using namespace std;

int main(){
	int n;
	cin >> n;
	while (n!=0){
	cout << bitset <8>(n) <<endl;
	cin >> n;
	}
	return 0; 
} 
