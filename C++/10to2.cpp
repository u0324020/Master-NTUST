#include <iostream>
#include <bitset>
//�Q�i����G�i��(�p�G��J0�Y����) 
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
