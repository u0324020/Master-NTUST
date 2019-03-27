#include <iostream>
using namespace std;
void Regulartriangle(int inx, int *starts) 
{	//タà 
	for(int s=0;s<=inx;s++){//北逼计 
		for(int k=(inx-s);k>=1;k--){//ぶフ = 羆︽计-ヘ玡︽计 
			cout << " "; }
		for(int j=0;j<s;j++){//–逼ぶ琍琍 
			cout<<"*";
			*starts+=1;
			}
		cout<<endl;//Ч︽传︽
	} 
		
}
void Invertedtriangle(int inx, int *starts) 
{	//à 
	for(int i=inx;i>0;i--){ //北逼计 
		for(int j=0;j<i;j++){ //北–逼琍琍计 
			cout<<"*";
			*starts+=1;
		}
		cout<<"\n"; //Ч︽传︽ 
	}	
}

int main(){
	int inx ,px ,starts=0,total=0;
	cin>>inx;
	while(inx >0){
	cin>>px;
	switch(inx){
		case 1  :
			Regulartriangle(px, &starts);
			printf("starts:%d\n",starts);
			total+=starts;	starts = 0;
			break; 
		case 2  :
			Invertedtriangle(px, &starts);
			printf("starts:%d\n",starts);
			total+=starts;	starts = 0;
			break; 
		default : 
    		break;
	}
	cin >> inx ;
	}
	printf("total starts:%d\n",total);
	system("pause");

}


