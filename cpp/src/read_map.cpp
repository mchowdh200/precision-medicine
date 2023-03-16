#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iterator>

#include "read_map.h"

using namespace std;

map<int, vector<tuple<int, float>>> map_bp_cm(
		string interpolated_map_file){
    ifstream file(interpolated_map_file);
    if (!file.is_open()){
        cout << "Error opening file: " << interpolated_map_file << endl;
        exit(1);
    }

    int curr_chrm = 0;

    string line;
    map<int, vector<tuple<int, float>>> bp_cm_map;
    vector<tuple<int, float>> single_chrm_bp_cm;

    while (getline(file, line)){
        istringstream iss(line);
        vector<string> tokens{istream_iterator<string>{iss}, istream_iterator<string>{}};

        int chrm = stoi(tokens[0]);
        int bp = stoi(tokens[2]);
        float cm = stof(tokens[1]);

        single_chrm_bp_cm.emplace_back(bp, cm);

        if (curr_chrm == 0){
            curr_chrm = chrm;
        }
        if (chrm != curr_chrm){
            bp_cm_map[curr_chrm] = single_chrm_bp_cm;
            curr_chrm = chrm;
            single_chrm_bp_cm.clear();
            single_chrm_bp_cm.emplace_back(bp, cm);
        }

    }
    bp_cm_map[curr_chrm] = single_chrm_bp_cm;
    return bp_cm_map;
}
