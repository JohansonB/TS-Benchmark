#include <iostream>
#include "TRMF.h"
#include "tron.h"

int main(int argc, char* argv[]) {
	trmf_param_t param = trmf_param_t();
	param.lambdaI = atof(argv[1]);
	param.lambdaAR = atof(argv[2]);
	param.lambdaLag = atof(argv[3]);
	size_t RANK = atoi(argv[4]);
	int nb_window = atoi(argv[5]);
	int window = atoi(argv[6]);
	char* data_path = argv[7];
	int ar_order = atoi(argv[8]);
	int stride = atoi(argv[9]);



	arma::uvec lagset(ar_order);
	if (stride > 1) {
		lagset(0) = 1;
	}
	int off_set = stride > 1 ? 0 : 1;
	int index;
	for (int i = 1-off_set; i < ar_order; i++) {
		index = (i + off_set) * stride;
		std::cout << index;
		lagset(i) = index;
	}

	arma::mat data;
	data.load(data_path);
	data = data.t();

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

	auto t1 = high_resolution_clock::now();

	arma::mat pred;
	pred = multi_pred(data, param, window, nb_window, lagset, RANK);
	pred = pred.t();
	

	auto t2 = high_resolution_clock::now();

	duration<double, std::milli> ms_double = t2 - t1;
	double s = ms_double.count() / 1000;

	
	//std::cout << pred;
	std::ofstream myfile;
	myfile.open("Scripts\\C\\Output\\TRMF_runtime.txt");
	myfile << s;
	myfile.close();

	pred.save("Scripts\\C\\Output\\TRMF_out.csv", arma::csv_ascii);

}



