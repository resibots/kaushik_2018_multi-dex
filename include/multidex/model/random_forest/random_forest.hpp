#ifndef LIMBO_MODEL_RF_HPP
#define LIMBO_MODEL_RF_HPP

#include <cassert>
#include <iostream>
#include <limbo/tools.hpp>

#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

namespace multidex {
    namespace defaults {
        struct model_randomForest{
            BO_PARAM(int, max_depth, 20);
            BO_PARAM(double, regression_accuracy, 0.001);
            BO_PARAM(int, nactive_vars, 0);
            BO_PARAM(int, max_trees, 1000);
            BO_PARAM(double, forest_accuracy, 0.00001);  
        };   
    }//namespace defaults
    namespace model {

        template <typename RFParams>
        class RandomForest {
        public:
            RandomForest() : _dim_in(-1), _dim_out(-1), _nb_samples(0) {}

            RandomForest(int dim_in, int dim_out) : _dim_in(dim_in), _dim_out(dim_out) { _nb_samples= 0;}

            void compute(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations)
            {
                _dim_in = samples[0].size();
                _dim_out = observations[0].size();
                _nb_samples = samples.size();

                Eigen::MatrixXd sampl(samples.size(), samples[0].size());
                Eigen::MatrixXd obs(observations.size(), observations[0].size());
                
                for (size_t i = 0; i < samples.size(); i++) {
                    sampl.row(i) = samples[i];
                    obs.row(i) = observations[i];
                }

                cv::Mat input, output;
                cv::eigen2cv(sampl, input);
                cv::eigen2cv(obs, output);

                input.convertTo(input, CV_32FC1);
                output.convertTo(output, CV_32FC1);

                cv::Mat var_type = cv::Mat(samples[0].size() + 1, 1, CV_8U);
                for (int i = 0; i < samples[0].size() + 1; i++)
                    var_type.at<uchar>(i, 0) = CV_VAR_ORDERED;

                int max_depth = RFParams::max_depth(); //NOTE: This parameter must be optimized
                int min_sample = std::max(1., 0.01 * samples.size()); // NOTE: Higer value underfits
                double regression_accuracy = RFParams::regression_accuracy(); //NOTE: When very small. Fitting is not very sensitive to this parameter. Keep around 0.0001
                int nactive_vars = 0;
                int max_num_of_trees_in_the_forest = RFParams::max_trees(); //NOTE: Higher the value smoter the curve. Make it hyperparameter
                double forest_accuracy = RFParams::forest_accuracy(); //NOTE: Make it a hyperparameter
                CvRTParams params(max_depth, min_sample, regression_accuracy, false, 2, NULL, false, nactive_vars, max_num_of_trees_in_the_forest, forest_accuracy, CV_TERMCRIT_ITER | CV_TERMCRIT_EPS);
                _forest.train(input, CV_ROW_SAMPLE, output, cv::Mat(), cv::Mat(), var_type, cv::Mat(), params);
            }

            std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& x) const
            {
                cv::Mat p;
                cv::eigen2cv(x, p);
                p.convertTo(p, CV_32FC1);

                double val = 0.;
                int N = _forest.get_tree_count();
                // std::vector<double> vals;
                Eigen::VectorXd vals(N);
                // for (int i = 0; i < N; i++) {
                limbo::tools::par::loop(0, N, [&](size_t i) {
                    // vals.push_back(_forest.get_tree(i)->predict(p)->value);
                    vals(i) = _forest.get_tree(i)->predict(p)->value;
                    // val += vals.back();
                });
                // val /= double(N);
                val = vals.mean();

                double sigma = 0.;

                for (int i = 0; i < N; i++) {
                    sigma += (vals[i] - val) * (vals[i] - val);
                }
                sigma /= double(N);

                // return tools::make_vector(_forest.predict(p));
                return std::make_tuple(limbo::tools::make_vector(val), sigma);
            }

            Eigen::VectorXd mu(const Eigen::VectorXd& x) const
            {
                cv::Mat p;
                cv::eigen2cv(x, p);
                p.convertTo(p, CV_32FC1);

                double val = 0.;
                int N = _forest.get_tree_count();
                Eigen::VectorXd vals(N);
                limbo::tools::par::loop(0, N, [&](size_t i) {
                    vals(i) = _forest.get_tree(i)->predict(p)->value;
                });
                val = vals.mean();
                return limbo::tools::make_vector(val);
            }

            double sigma(const Eigen::VectorXd& x) const
            {
                cv::Mat p;
                cv::eigen2cv(x, p);
                p.convertTo(p, CV_32FC1);

                double val = 0.;
                int N = _forest.get_tree_count();
                // std::vector<double> vals;
                Eigen::VectorXd vals(N);
                // for (int i = 0; i < N; i++) {
                limbo::tools::par::loop(0, N, [&](size_t i) {
                    // vals.push_back(_forest.get_tree(i)->predict(p)->value);
                    vals(i) = _forest.get_tree(i)->predict(p)->value;
                    // val += vals.back();
                });
                // val /= double(N);
                val = vals.mean();

                double sigma = 0.;

                for (int i = 0; i < N; i++) {
                    sigma += (vals[i] - val) * (vals[i] - val);
                }
                sigma /= double(N);

                // return tools::make_vector(_forest.predict(p));
                return sigma;
            }

            void optimize_hyperparams()
            {
                std::cout<<"RF samples : "<< _nb_samples<<std::endl;
            }

            int get_tree_count()
            {
                return _forest.get_tree_count();
            }

            size_t nb_samples()
            {
                return _nb_samples;
            }

        protected:
            CvRTrees _forest;
            int _dim_in, _dim_out;
            size_t _nb_samples;
        };
        
        template <typename RFParams>
        class MultiRandomForest{
        public:
            MultiRandomForest() : _dim_in(-1), _dim_out(-1),  _nb_samples(0) {}

            MultiRandomForest(int dim_in, int dim_out) : _dim_in(dim_in), _dim_out(dim_out),  _nb_samples(0) {}

            void compute(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations)
            {
                _dim_in = samples[0].size();
                _dim_out = observations[0].size();
                _forests.resize(_dim_out);
                _nb_samples = samples.size();

                for(size_t i = 0; i < _dim_out; i++ )
                {
                    std::vector<Eigen::VectorXd> obs(samples.size());
                        
                    for(size_t j=0; j< samples.size(); j++)
                    {
                        obs[j] = limbo::tools::make_vector(observations[j](i));
                    }
                    _forests[i].compute(samples, obs);
                }   
            }

            std::tuple<Eigen::VectorXd, Eigen::VectorXd> query(const Eigen::VectorXd& x) const
            {
                Eigen::VectorXd mean(_dim_out);
                Eigen::VectorXd var(_dim_out);
                for(size_t i=0; i < _dim_out; i++ )
                {
                    
                    Eigen::VectorXd m;
                    std::tie(m,var[i])  = _forests[i].query(x);
                    mean[i] = m[0];
                }

                return std::make_tuple(mean,var);
            }

            Eigen::VectorXd mu(const Eigen::VectorXd& x) const
            {
                Eigen::VectorXd mean(_dim_out);
                // Eigen::VectorXd var(_dim_out);
                for(size_t i=0; i < _dim_out; i++ )
                {
                    
                    Eigen::VectorXd m;
                    m = _forests[i].mu(x);
                    mean[i] = m[0];
                }

                return mean;
            }

            Eigen::VectorXd sigma(const Eigen::VectorXd& x) const
            {
                Eigen::VectorXd mean(_dim_out);
                Eigen::VectorXd var(_dim_out);
                for(size_t i=0; i < _dim_out; i++ )
                {
                    
                    Eigen::VectorXd m;
                    std::tie(m,var[i])  = _forests[i].query(x);
                    mean[i] = m[0];
                }

                return var;
            }

             /// return the number of dimensions of the input
            int dim_in() const
            {
                assert(_dim_in != -1); // need to compute first !
                return _dim_in;
            }

            /// return the number of dimensions of the output
            int dim_out() const
            {
                assert(_dim_out != -1); // need to compute first !
                return _dim_out;
            }
            
            void optimize_hyperparams()
            {}

            size_t nb_samples()
            {
                return _nb_samples;
            }

        protected:
            int _dim_in, _dim_out;
            std::vector<RandomForest<RFParams>> _forests;
            size_t _nb_samples;
        };
    } // namespace model
} // namespace multidex

#endif