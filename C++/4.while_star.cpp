#include <iostream>
using namespace std;
//タà
void RegularTriangle(int n){
	int s=0; 
	while(s<n){//北逼计 
		s+=1;
		int k=(n-s);
		while(k>=1){//ぶフ = 羆︽计-ヘ玡︽计 
			k-=1;
			cout << " "; }
			int j=0;
		while(j<s){//–逼ぶ琍琍 
			j+=1;
			cout<<"*";}
		cout<<endl;}
		cout<<endl;//Ч︽传︽ 
}
//à 
void InvertedTriangle(int n){
	int i=n+1;
	while(i>0){ //北逼计 
		i-=1;
		int j=0;
		while(j<i){ //北–逼琍琍计 
			j+=1;
			cout<<"*";
		}
		cout<<"\n"; //Ч︽传︽ 
	}
}
int main(){  
	int n;
	cin>>n;
	RegularTriangle(n);
	InvertedTriangle(n);
	system("pause");
	return 0;
}

