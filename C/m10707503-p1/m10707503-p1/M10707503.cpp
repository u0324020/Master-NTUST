#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <vector>
using namespace std;
struct point{
	int Point_1;
    int Point_2;};
point chord;
vector<point> out;
vector<point> table;    
vector<vector<int> > Case;
vector<vector<int> > Case_try;
vector<vector<int> > MPSC;
vector<int> k(10000);
int C_num =0;
int Max_num=0;
int Number=0;
int N = 0;

int BestAnswer(int i,int j){ // 本段function參考網路資源
  while(i<j){ 
      switch(Case_try[i][j]){  
        case 1:
	        Number=1;
	        j--;
	        break;
        case 2:
	        Number=2;
	        j--;
	        break;
        case 3: 
	        chord.Point_1=i;
	        chord.Point_2=j;
	        out.push_back(chord);
	        C_num=C_num+1;
	        Number=3;
	        i++;
	        j--; 
	        break;
        case 4:
	        chord.Point_1=k[j];
	        chord.Point_2=j;
	        out.push_back(chord);
	        C_num=C_num+1;
	        chord.Point_1=i;
	        chord.Point_2=k[j];
	        table.push_back(chord);
	        Number=4;
	        i=k[j]+1;
	        j--;
	        break;
        default:
	        Number=5;
	        break;}}}

int main(int argc, char **argv)   
{
    if(argc != 3)
    {
        cout << "Usage: ./[exe] [input file] [output file]" << endl;
        system("pause");
    }

    // open the input file
    fstream fin;
    fin.open(argv[1], fstream::in);
    if(!fin.is_open())
    {
        cout << "Error: the input file is not opened!!" << endl;
        exit(1);
    }

    char buffer[10000];
    fin >> buffer;
    int numChord = atoi(buffer);
    N = numChord;
    int x = 0;
	int y = N-1;//0-11*0-11
    k.resize(10000);
	Case.resize(10000);
	MPSC.resize(10000);
	Case_try.resize(10000);
	
	for(int w=0;w<10000;++w){
		Case[w].resize(10000);
		MPSC[w].resize(10000);
		Case_try[w].resize(10000);}
		
    for(int z=0; z<N/2;++z){//===================for loop one=====================
	    fin >> buffer;
	    int point1 = atoi(buffer);
	    fin >> buffer;
	    int point2 = atoi(buffer);
	    for (int i=0;i<=N-1;++i){//0-11
			for(int j=0;j<=N-1;++j){//0-11                                   
			    if(j==point1){
		    		k[j]=point2;  //Chord point
		    		if (k[j]>j||k[j]<i){//find Case
						Case[i][j]=1;}
		    		else if(k[j]<j&&k[j]>i){
		    			Case[i][j]=2;}
		    		else if (k[j] == i){
						Case[i][j]=3;}}
		     	if(j==point2){  
		    		k[j]=point1;
		     		if (k[j]>j||k[j]<i){
		    			Case[i][j]=1;}
		    		else if(k[j]<j&&k[j]>i){
		       			Case[i][j]=2;}
			     	else if (k[j] == i){
			       		Case[i][j]=3;}}}}}
			       		
	for(int j= 1;j<=N-1;++j){//===========================for loop two=============================
		for(int i=0;i<j;++i){   
			switch(Case[i][j]){
				case 1:
					MPSC[i][j]=MPSC[i][j-1];
					Case_try[i][j]=1;//Case 1
					break;
				case 2:
					MPSC[i][j]=max((MPSC[i][j-1]),(MPSC[i][k[j]-1]+1+MPSC[k[j]+1][j-1]));             
					if(MPSC[i][j-1] > (MPSC[i][k[j]-1]+1+MPSC[k[j]+1][j-1])){
						Case_try[i][j]=2;//Case 2-1
					}else{
						Case_try[i][j]=4;//Case 2-2
					}break;
				case 3:
					MPSC[i][j]=MPSC[i+1][j-1]+1;
					Case_try[i][j]=3;//Case 3
					break;
				default:
					break;}}}
					
 int Max_num=MPSC[0][N-1];
//
//================================Max Answer========================================
 BestAnswer(x,y);
 for(int i=0;i<table.size();++i){
   BestAnswer(table[i].Point_1,table[i].Point_2);}
   //cout<<"LOOP ANS"<<endl;
 // open the output file
    fstream fout;
    fout.open(argv[2],fstream::out);
    if(!fout.is_open())
    {
        cout << "Error: the output file is not opened!!" << endl;
     exit(1);
    }
  
for(int i=0; i < out.size()-1 ; ++i){//chord number
	for(int j=i+1 ; j<out.size() ; ++j ){
		if(out[j].Point_1<out[i].Point_1){
			swap(out[i] , out[j]);
		}}}
cout<<out.size()<<endl;
for(int i=0; i < out.size(); ++i){
	cout<<out[i].Point_1<<" "<<out[i].Point_2<<endl;}
    system("pause");
    return 0;}

