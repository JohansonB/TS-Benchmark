#include <iostream>
#include "TRMF.h"
#include "tron.h"

/*int main(int argc, char* argv[]) {
	trmf_param_t param = trmf_param_t();
	param.lambdaI = atof(argv[1]);
	param.lambdaAR = atof(argv[2]);
	param.lambdaLag = atof(argv[3]);
	size_t RANK = atoi(argv[4]);

	arma::uvec lagset;
	if (argc == 7) {
		int len = atoi(argv[6]);
		lagset = arma::uvec(len);
		for (int i = 0; i < len; i++) {
			lagset[i] = i + 1;
		}
	}
	else {
		lagset << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 9 << 10 << 11 << 12 << 13 << 14 << 15 << 16 << 17 << 18 << 19 << 20 << 21 << 22 << 23 << 24 << 168 << 169 << 170 << 171 << 172 << 173 << 174 << 175 << 176 << 177 << 178 << 179 << 180 << 181 << 182 << 183 << 184 << 185 << 186 << 187 << 188 << 189 << 190 << 191;
	}

	arma::mat data;
	data.load(argv[5]);
	data = data.t();

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

	auto t1 = high_resolution_clock::now();

	
	std::tuple<arma::mat,arma::mat,arma::mat> model = return_model(data, param, lagset, RANK);
	


	auto t2 = high_resolution_clock::now();

	duration<double, std::milli> ms_double = t2 - t1;
	double s = ms_double.count() / 1000;



	std::ofstream myfile;
	myfile.open("Scripts\\C\\Output\\TRMF_runtime.txt");
	myfile << s;
	myfile.close();

	std::get<0>(model).save("Scripts\\C\\TRMF\\latent_series.csv", arma::csv_ascii);
	std::get<1>(model).save("Scripts\\C\\TRMF\\lag_values.csv", arma::csv_ascii);
	std::get<2>(model).save("Scripts\\C\\TRMF\\recombiner.csv", arma::csv_ascii);
	

}*/
