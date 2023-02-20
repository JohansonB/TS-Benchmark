
#include <math.h>
#include "OTS_ogd.h"

OTS_ogd::OTS_ogd(arma::mat &data, size_t ar_order, double eta) :
	TemplateOTS(data, eta), m_order(ar_order) {
	m_name = "ots_ogd";
	weights = arma::mat(m_M, ar_order+1);
	m_alpha = arma::mat(m_N, ar_order + 1, arma::fill::zeros);

}
using arma::reverse;
using arma::as_scalar;
arma::vec OTS_ogd::vec_forecast(arma::vec const &to_pred) {

	arma::colvec pred = to_pred;

	for (size_t i = 0; i < m_order + 1; ++i) {
		m_err(i) = pow(pred(i), 2);
	}

	static arma::vec one(1, arma::fill::ones);
	for (size_t t = m_order; t < m_N; t++) {

		if (isnan(to_pred(t))) {

			arma::colvec temp = reverse(pred.rows(t - m_order, t - 1));
			temp.insert_rows(0, one.row(0));

			std::cout << "temp" << temp << "m_alpha" << m_alpha.row(t - 1) << "\n";
			

			pred(t) = as_scalar(m_alpha.row(t - 1) * temp);

			std::cout << "pred" << pred(t)<<"\n";

		}

		if (not isnan(to_pred(t))) {

			arma::colvec temp = reverse(pred.rows(t - m_order, t - 1));
			temp.insert_rows(0, one.row(0));
			//pred(t) = as_scalar(m_alpha.row(t - 1) * temp);
			m_err(t) = as_scalar(to_pred(t) - as_scalar(m_alpha.row(t - 1) * temp));

		}

		arma::colvec temp = reverse(pred.rows(t - m_order, t - 1));
		temp.insert_rows(0, one.row(0));

		arma::rowvec gradient = 2 * (as_scalar(m_alpha.row(t - 1) * temp) - pred(t)) * temp.t();

		if (!isnan(to_pred(t))) {
			m_alpha.row(t) = m_alpha.row(t - 1) - (m_eta / sqrt(t)) * gradient;
			m_alpha.row(t) = m_alpha.row(t) / fmax(1., sqrt(arma::accu(arma::pow(m_alpha.row(t), 2))));
		}
		else {
			m_alpha.row(t) = m_alpha.row(t - 1);
		}


	}
	weights.row(count++) = m_alpha.row(m_N - 1);


	return pred;
}