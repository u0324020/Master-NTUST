#include <iostream>
using namespace std;
//计禲皑縊 
int main(){ 
	int jc = 0;
	for(int i=0;i<=30;i++){ //北羆计秖 
		cout << "\r"; //–Ω程玡よ 
		for(int k=(30-i);k>=1;k--){//惠ぶフ 
			cout << " "; }
		for(int j=1;j<=i;j++){//计 
			cout << j%10;}
		jc=1;
		while(1){//丁┑筐 
			if(jc>50000000) break;
			jc++;}}
		cout<<endl;
		system("pause");
		return 0;
}


