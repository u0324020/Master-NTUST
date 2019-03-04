#include <iostream>
#include <bitset>
//十進制轉二進制(1.輸入次數 3.輸入轉制數值(介於-127~128)) 
using namespace std;

int main(){
	int n;
	int k;
	cin >> k;
	for(int i=0;i<k;i++){
		cin >> n;
		if (-128<=n && n<=127){
		cout << bitset <8>(n) <<endl;
		}else{cout << "Error Input !\n";
		}
	}
	cout <<"Finish !\n";
	
	return 0; 
} 

