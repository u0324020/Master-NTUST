#include <iostream>
using namespace std;
void Regulartriangle(int inx, int *starts) 
{	//���T���� 
	for(int s=0;s<=inx;s++){//����Ƽ� 
		for(int k=(inx-s);k>=1;k--){//�L�h�֪ť� = �`���-�ثe��� 
			cout << " "; }
		for(int j=0;j<s;j++){//�C�Ʀh�֬P�P 
			cout<<"*";
			*starts+=1;
			}
		cout<<endl;//�L���@��Y����
	} 
		
}
void Invertedtriangle(int inx, int *starts) 
{	//�ˤT���� 
	for(int i=inx;i>0;i--){ //����Ƽ� 
		for(int j=0;j<i;j++){ //����C�ƬP�P�� 
			cout<<"*";
			*starts+=1;
		}
		cout<<"\n"; //�L���@��Y���� 
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


