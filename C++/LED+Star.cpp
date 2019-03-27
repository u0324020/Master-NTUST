#include <iostream>
using namespace std;
//數字跑馬燈 + 印星星 
int main(){ 
	int jc;
	for(int s=10;s>=0;s--){//控制排數 
		for(int i=0;i<=s;i++){//幾顆星星即返回幾次 
			cout << "\r"; 
			for(int k=(s-i);k>=1;k--){//應印多少空白 
				cout << " "; }
			for(int j=1;j<=i;j++){//應印多少星星 
				cout << "*";}
			jc=1;
			while(1){//延遲 
				if(jc>20000000) break;
				jc++;
			}}
			cout<<endl;} 
		system("pause");
		return 0;}
		
		

