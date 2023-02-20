#include <iostream>
#include <fstream>
#include <string> 
#include <armadillo>
#include <chrono>
#include "OTS_gsr.h"
#include "OTS_ogd.h"
using namespace std;
using namespace arma;

/*int main(int argc, char* argv[])
{
	int order = atoi(argv[1]);
	double eta = atof(argv[2]);
	int algo = atoi(argv[3]);
	string dataPath = argv[4];

	
	mat data;
	data.load(dataPath, csv_ascii);
	data = data.t();

	TemplateOTS* alg;
	if (algo == 0) {
		alg = new OTS_ogd(data,order, eta);
	}
	else {
		alg = new OTS_gsr(data, order, eta);
	}


	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

	auto t1 = high_resolution_clock::now();


	mat pred = alg->forecast();

	auto t2 = high_resolution_clock::now();

	duration<double, std::milli> ms_double = t2 - t1;
	double s = ms_double.count() / 1000;

	pred = pred.t();

	ofstream myfile;
	myfile.open("Scripts\\C\\Output\\OTS_runtime.txt");
	myfile << s;
	myfile.close();


	
	pred.save("Scripts\\C\\Output\\OTS_out.csv", csv_ascii);
	
	//test.save("C:\\Users\\41766\\IdeaProjects\\BrigidJones\\Scripts\\C\\Output\\LSRN_out.csv", csv_ascii);




}*/