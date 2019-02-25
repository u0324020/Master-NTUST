#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <omp.h>
#include <vector>
#include <fstream>
#include <time.h> 

//If you want using serial, Please comment the following 3 lines.
#define thread1
#define thread2
#define MergeThread

using namespace std;
vector<int> A, B;
void mergeSort(vector<int>&left, vector<int>& right, vector<int>& bars);
void sort(vector<int> & bar, int);

void sort(vector<int> & bar, int threads) {
    if (bar.size() <= 1) return;

    int mid = bar.size() / 2;
    vector<int> left;
    vector<int> right;

    for (size_t j = 0; j < mid;j++)
        left.push_back(bar[j]);
    for (size_t j = 0; j < (bar.size()) - mid; j++)
        right.push_back(bar[mid + j]);
#ifdef thread1
	#pragma omp parallel sections num_threads(threads)
#endif thread1
	{
#ifdef thread1
	#pragma omp section
#endif thread1
	{
	#ifdef DEBUGThread
	printf( "L, %d", omp_get_thread_num() );	
	#endif DEBUGThread
	sort(left,threads);		}}
#ifdef thread2
	#pragma omp parallel sections num_threads(threads)
#endif thread2
	{
#ifdef thread2
	#pragma omp section
#endif thread2
	{
	#ifdef DEBUGThread
	printf( "R, %d", omp_get_thread_num() );	
	#endif DEBUGThread
		sort(right,threads);	}
	}
#ifdef MergeThread 
	#pragma omp parallel sections num_threads(threads) 
#endif MergeThread 
	{
#ifdef MergeThread 
	#pragma omp section  
#endif MergeThread 
	{ 
	mergeSort(left, right, bar);
#ifdef DEBUGMerge
	printf( "M, %d", omp_get_thread_num() );	
#endif DEBUGMerge
	} 
	}
}
void mergeSort(vector<int>&left, vector<int>& right, vector<int>& bars)
{
    int nL = left.size();
    int nR = right.size();
    int i = 0, j = 0, k = 0;

    while (j < nL && k < nR) 
    {
        if (left[j] < right[k]) {
            bars[i] = left[j];
            j++;
        }
        else {
            bars[i] = right[k];
            k++;
        }
        i++;
    }
    
    while (j < nL) {
        bars[i] = left[j];
        j++; i++;
    }
    while (k < nR) {
        bars[i] = right[k];
        k++; i++;
    }
}


int main(int argc, char **argv)
{
cout<<"**************************************"<<endl;
cout<<"      MERGE SORT PROGRAM       "<<endl;
cout<<"**************************************"<<endl;

cout<<endl<<endl;
cout<<endl;
    // Check arguments
    if (argc != 4)		/* argc must be 3 for proper execution! */
    {
        printf ("Usage: [exe] [#threads] [case.in] [case.out]\n", argv[0]);
        return 1;
    }

    // Get arguments
    int threads = atoi(argv[1]);	// Requested number of threads

    // Check nested parallelism availability
    omp_set_nested(1);
    if (omp_get_nested () != 1)
    {
        puts ("Warning: Nested parallelism desired but unavailable");
    }

    // Check threads
    int max_threads = omp_get_max_threads ();	// Max available threads
    if (threads > max_threads)	// Requested threads are more than max available
    {
        printf ("Error: Cannot use %d threads, only %d threads available\n",
                threads, max_threads);
        return 1;
    }

    // parse input
    fstream fin;
    fin.open(argv[2], fstream::in);
    if( !fin.is_open() )
    {
        printf("case.in is not opened!\n");
        return 1;
    }

    char buffer[5];
    fin >> buffer; // vector size
    int vec_size = atoi(buffer);
    // vector allocation
    A.resize(vec_size);
    B.resize(vec_size, -1);
    for( int i = 0; i < (int)A.size(); ++ i )
    {
        fin >> buffer;
        A[i] = atoi(buffer);
    }

    //********************************
    //***** start your code here *****
    //********************************



sort(A,threads);
cout<<endl;
cout<<"So, the sorted list (using MERGE SORT) will be :"<<endl;
cout<<endl<<endl;

#ifdef Print
for(int i=1;i<=vec_size;i++)
	cout<<A[i]<<"  ";
#endif Print

cout<<endl<<endl<<endl<<endl;

    printf("start multi-threaded merge sort......\n");

    //write sorted result into case.out
    fstream fout;
    fout.open(argv[3], fstream::out);
    if( !fout.is_open() )
    {
        printf ("case.out is not opened!\n");
        return 1;
    }

	//Write result
    for ( int i = 0; i < vec_size; ++ i )
    {   //printf("%d ", temp[i]);
        fout << A[i] << endl;
    }
    
cout << (double)clock() / CLOCKS_PER_SEC << " S"<<endl;
    printf("Multi-threaded merge sort succeed.\n");
    return 0;
}


