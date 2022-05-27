#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>

#include "bruteForce.h"
#include "readEncoding.h"

using namespace std;

int brute_force_main(string encodedFile, string queriesFile, int numVariants, int numSamples, int numQueries){
	
	// get queries from file
        float* queries = read_queries(queriesFile, numVariants, numQueries);
	for(int q = 0; q < numQueries; q++){

		// one query at a time
		float* currQuery = new float[numVariants];
		for(int i = 0; i < (numVariants); i++){
			currQuery[i] = queries[q * numVariants + i];
		}
		cout << "Query " << q << endl; 
		float* distArr = compute_one_query(currQuery, encodedFile, numVariants, numSamples, numQueries);

		delete[] currQuery;
	}
	return 0;
}

float *compute_one_query(float* query, string encodedFile, int numVariants, int numSamples, int numQueries){
	// ifstream to encoded file
        ifstream inFile;
        // open encoded file
        inFile.open(encodedFile);
        if ( !inFile.is_open() ) {
                cout << "Failed to open: " << encodedFile << endl;
        }

	// store index and distance in my own hashmap
	float* distArr = new float[numSamples];
        
	// read encoded file line by line
        string line;
        int lineCount = 0;
        if(inFile.is_open()){
                while(getline(inFile, line)){
                        int segLength = line.length();
                        if (segLength != numVariants){
                                cerr << "\t!! segment length does not equal number of variants !!" << endl;
                        }
                        string s;
                        float f;

			// convert string line to float array
                        float* singleVector = new float[segLength];
                        for (int c = 0; c < segLength; c++){
                                s = line[c];
                                f = stof(s);
                                singleVector[c] = f;
                        }
			float singleDistance = euclidean_distance(query, singleVector, segLength);
			distArr[lineCount] = singleDistance;
			lineCount ++;
			/*
			cout << "\tvector" << lineCount << ": " << distance << endl;
			*/

		}
	}
	cout << "...brute force computations complete." << endl;
	return distArr;
}


float euclidean_distance(float* vec1, float* vec2, int segLength){

	/*
	for (int i = 0; i < segLength; i++)
                cout << vec1[i] << " ";
	cout << endl;
	for (int i = 0; i < segLength; i++)
                cout << vec2[i] << " ";
	cout << endl;
	*/
	float eucDist = 0;
	float sum = 0;
	for (int i = 0; i < segLength; i++){
		float diff = vec1[i]-vec2[i];
		float diffSqrd = pow(diff, 2);
		sum += diffSqrd;
	}
	eucDist = sqrt(sum);
	return eucDist;
}
