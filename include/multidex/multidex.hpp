#ifndef MULTIDEX_HPP
#define MULTIDEX_HPP

#include <limits>
#include <queue> 
#include <Eigen/binary_matrix.hpp>

namespace multidex 
{
    template < typename Params, typename Policy, typename RewardFunction, typename Robot, typename DynamicsModel>
    class Multidex 
    {
        using obsDynamics_t = std::vector<std::tuple<Eigen::VectorXd,Eigen::VectorXd,Eigen::VectorXd, Eigen::VectorXd>>;
        
    public:
        Policy _policy;
        Robot _robot;
        DynamicsModel _model;
        RewardFunction _reward_func;
        std::vector<Eigen::VectorXd> _obs_behaviors;  
        std::vector<Eigen::VectorXd> _expected_behaviors;  
        int _opt_iters;
        size_t _steps;
        size_t _data_limit;
        double _best_reward;
        double _last_reward;
        std::vector<double> _all_step_rewards;
        std::vector<double> _best_step_rewards;
        std::vector<double> _best_accumulated_rewards;
        obsDynamics_t _obs_dynamics;
        obsDynamics_t _best_obs_dynamics; 
        obsDynamics_t _normal_obs_dynamics; 
        obsDynamics_t _limited_obs_dynamics;
        obsDynamics_t _limited_best_dynamics;
        double _last_novelty;
        Eigen::VectorXd _last_behavior;
        bool _policy_set;
        std::mutex _mutex;
        Eigen::VectorXd _max_predictions;
        std::vector<Eigen::VectorXd> _executed_policies;
        Eigen::VectorXd _best_policy_params;
        std::vector<Eigen::VectorXd> _model_inputs; //for mean model
        std::vector<Eigen::VectorXd> _model_outputs; //For mean model

    public:
        Multidex() 
        {
            _best_reward = -std::numeric_limits<float>::max();
            _data_limit = Params::multidex::model_data_limit();
            _policy_set = false;
            _steps = size_t(Params::multidex::T()/Params::multidex::dt());

            /** Update the maximum prediction limits by the dynamics model*/
            _max_predictions = Eigen::VectorXd::Zero(Params::multidex::model_pred_dim());                      
            for (int i = 0; i < _max_predictions.size(); i++) {
                _max_predictions(i) = std::abs(Params::multidex::model_pred_limits(i));
            }
        }

        void execute_random_policy(const size_t& ep=0)
        {
            /** --------------------------------------------------------------------------- 
             * Instanciate everything 
             */
            RewardFunction rewFunc;
            std::vector<double> R ;
            Robot robot;
            Policy policy;

            /** ---------------------------------------------------------------------------- 
             * Execute random Policy on the robot */ 
            policy.set_random_policy();
            _policy_set = true;
            _executed_policies.push_back(policy.params());
            bool collision = false;
            auto obs_new = robot.execute(policy, rewFunc, Params::multidex::T(), R, collision);            
            std::cout<<"Random policy executed.."<<std::endl;

            for (size_t i=0; i < obs_new.size(); i++)
            {
                Eigen::VectorXd input(Params::multidex::model_input_dim() + Params::multidex::action_dim());
                Eigen::VectorXd output(Params::multidex::model_pred_dim());

                input.head(Params::multidex::model_input_dim()) = std::get<0>(obs_new[i]);
                input.tail(Params::multidex::action_dim()) = std::get<1>(obs_new[i]);
                output = std::get<2>(obs_new[i]);
                _model_inputs.push_back(input);
                _model_outputs.push_back(output);
            }

            /** ------------------------------------------------------------------------------ 
             * store reward and dynamics 
              */
            _all_step_rewards.insert(_all_step_rewards.end(), R.begin(), R.end());
            _last_reward = std::accumulate(R.begin(), R.end(), 0.0);
            
            if(_last_reward > _best_reward)
            {
                _best_reward = _last_reward;
                _best_policy_params = policy.params();
            }
           _obs_dynamics.insert(this->_obs_dynamics.end(), obs_new.begin(), obs_new.end());   

            /** ---------------------------------------------------------------------------------
             * Limit observations for model learning 
             */
            if(_last_reward > Params::multidex::define_good_rew())
            {
                std::cout<<"Good !! Saving in Best-Experience buffer"<<std::endl;
                _best_obs_dynamics.insert(_best_obs_dynamics.end(),obs_new.begin(),obs_new.end());
                _best_step_rewards.insert(_best_step_rewards.end(), R.begin(), R.end());
                _best_accumulated_rewards.push_back(_last_reward);
                
                if(_best_obs_dynamics.size() <= Params::multidex::model_data_limit())
                {
                    _data_limit = _best_obs_dynamics.size();
                }
                else
                {
                    _data_limit = Params::multidex::model_data_limit();
                }
                /** Match the FIFO experience buffer size with best exp buffer */
                _limited_obs_dynamics = limit_obsDynamics(_obs_dynamics);
               
            }
            else
            {
                std::cout<<"Saving in FIFO-Experience buffer"<<std::endl;
                _limited_obs_dynamics = limit_obsDynamics(_obs_dynamics);
                /** this is just to keep non rewarding trajects if we want to use**/
                _normal_obs_dynamics.insert(_normal_obs_dynamics.end(), obs_new.begin(), obs_new.end());
            }

            std::cout<<"Best exp buffer  : "<<_best_obs_dynamics.size()<<std::endl;
            std::cout<<"FIFO exp buffer  : "<<_limited_obs_dynamics.size()<<std::endl;
            // std::cout<<"Normal exp buffer: "<<_normal_obs_dynamics.size()<<std::endl;
            std::cout<<"data limit       : "<<_data_limit<<std::endl;
            /** --------------------------------------------------------------------------------
             * Archive executed and expected behaviors/trajectories
             * for novelty computation 
             */ 
            Eigen::VectorXd executed_behavior = this->compute_behavior(obs_new);
            this->_obs_behaviors.push_back(executed_behavior);
            this->_expected_behaviors.push_back(executed_behavior); /* Since right now no model*/
            
            /** ------------------------------------------------------------------------------ 
             * Store every step as a row in a matrix
             * and write into binary file
             */ 
            int columns = std::get<3>(obs_new[0]).size() + std::get<1>(obs_new[0]).size() + std::get<2>(obs_new[0]).size() + 1; //state, action, diffstate, reward
            Eigen::MatrixXd robot_execution_data(obs_new.size(), columns);
            for(size_t i=0; i< obs_new.size(); i++)
            {
                Eigen::VectorXd d(columns);
                d.segment(0,std::get<3>(obs_new[0]).size()) = std::get<3>(obs_new[i]);
                d.segment(std::get<3>(obs_new[0]).size(), std::get<1>(obs_new[0]).size()) = std::get<1>(obs_new[i]); 
                d.segment(std::get<3>(obs_new[0]).size() + std::get<1>(obs_new[0]).size(), std::get<2>(obs_new[0]).size()) = std::get<2>(obs_new[i]);
                d(d.size()-1) = R[i]; 
                robot_execution_data.row(i) = d; 
            }
            Eigen::write_binary("trajectory_data_"+ std::to_string(ep)+".bin", robot_execution_data);
            std::cout<<"Observations dumped: trajectory_data_"<< std::to_string(ep)<<".bin"<<std::endl;
      
            _last_novelty = -1;
            std::cout<<"Reward : "<<_last_reward<<std::endl;
            std::cout<<"Best Reward = "<<_best_reward<<std::endl;
            _policy_set = false;
        }

        void execute_robot(const size_t& ep)
        {
            //NOTE: Assumes that policy is set in _policy
            assert((_policy_set==true)&&("Can't execute the robot. First set the policy"));
            std::vector<double> R;
            bool collision = false;
            auto obs_new = _robot.execute(_policy, _reward_func, Params::multidex::T(), R, collision);
            _executed_policies.push_back(_policy.params());
            std::cout<<"Optimized Policy Executed.."<<std::endl;

            for (size_t i=0; i < obs_new.size(); i++)
            {
                Eigen::VectorXd input(Params::multidex::model_input_dim() + Params::multidex::action_dim());
                Eigen::VectorXd output(Params::multidex::model_pred_dim());

                input.head(Params::multidex::model_input_dim()) = std::get<0>(obs_new[i]);
                input.tail(Params::multidex::action_dim()) = std::get<1>(obs_new[i]);
                output = std::get<2>(obs_new[i]);
                _model_inputs.push_back(input);
                _model_outputs.push_back(output);
            }
            
            _last_reward = std::accumulate(R.begin(), R.end(), 0.0);
            
            bool keep_data = (!collision) || (_last_reward > Params::multidex::define_good_rew()); 
            if (keep_data)
            {                                   
                /** ------------------------------------------------------------------------------ 
                 * store reward and dynamics 
                 */
                _all_step_rewards.insert(_all_step_rewards.end(), R.begin(), R.end());
                if(_last_reward > _best_reward)
                {
                    _best_reward = _last_reward;
                    _best_policy_params = _policy.params();
                }  
                _obs_dynamics.insert(_obs_dynamics.end(), obs_new.begin(), obs_new.end());      
                
                /** ---------------------------------------------------------------------------------
                 * Ensure best-experience data size = FIFO-Experience buffer size
                 */ 
                if(_best_obs_dynamics.size() > 0 && _best_obs_dynamics.size() <= Params::multidex::model_data_limit())
                    _data_limit = _best_obs_dynamics.size();
                else
                    _data_limit = Params::multidex::model_data_limit();                

                /** ---------------------------------------------------------------------------------
                 * Limit observations for model learning 
                 */
                if(_last_reward > Params::multidex::define_good_rew())
                {
                    std::cout<<"Good !! Saving in Best-Experience buffer"<<std::endl;
                    _best_obs_dynamics.insert(_best_obs_dynamics.end(),obs_new.begin(),obs_new.end());
                    _best_step_rewards.insert(_best_step_rewards.end(), R.begin(), R.end());
                    _best_accumulated_rewards.push_back(_last_reward);

                    /*NOTE: TRIAL: To match the original version*/
                    _obs_dynamics = _best_obs_dynamics;
                    _all_step_rewards = _best_step_rewards;
                    /*----trial---------------------------------*/ 

                    /** Match the FIFO experience buffer size with best exp buffer */
                    if(_best_obs_dynamics.size() <= Params::multidex::model_data_limit())
                    {
                        _data_limit = _best_obs_dynamics.size();
                    }
                    else
                    {
                        _data_limit = Params::multidex::model_data_limit();
                    }
                    _limited_obs_dynamics = limit_obsDynamics(_obs_dynamics);
                }
                else
                {
                    std::cout<<"Saving in FIFO-Experience buffer"<<std::endl;
                    _limited_obs_dynamics = limit_obsDynamics(_obs_dynamics);
                    /** just to keep non rewarding trajects if we want to use them **/
                    _normal_obs_dynamics.insert(_normal_obs_dynamics.end(), obs_new.begin(), obs_new.end());
                }                
                
                /** ------------------------------------------------------------------------------ 
                 * Store every step as a row in a matrix
                 * and write into binary file
                 */ 
                int columns = std::get<3>(obs_new[0]).size() + std::get<1>(obs_new[0]).size() + std::get<2>(obs_new[0]).size() + 1; //state, action, diffstate, reward
                Eigen::MatrixXd robot_execution_data(obs_new.size(), columns);
                for(size_t i=0; i< obs_new.size(); i++)
                {
                    Eigen::VectorXd d(columns);
                    d.segment(0,std::get<3>(obs_new[0]).size()) = std::get<3>(obs_new[i]);
                    d.segment(std::get<3>(obs_new[0]).size(), std::get<1>(obs_new[0]).size()) = std::get<1>(obs_new[i]); 
                    d.segment(std::get<3>(obs_new[0]).size() + std::get<1>(obs_new[0]).size(), std::get<2>(obs_new[0]).size()) = std::get<2>(obs_new[i]);
                    d(d.size()-1) = R[i]; 
                    robot_execution_data.row(i) = d; 
                }
                Eigen::write_binary("trajectory_data_"+ std::to_string(ep)+".bin", robot_execution_data);
                std::cout<<" Observations dumped: trajectory_data_"<< std::to_string(ep)<<".bin"<<std::endl;
            }
            else
            {
                std::cout<<"Bad dynamics. NOT storing dynamics !!"<<std::endl;
            }

            std::cout<<"Best exp buffer  : "<<_best_obs_dynamics.size()<<std::endl;
            std::cout<<"FIFO exp buffer  : "<<_limited_obs_dynamics.size()<<std::endl;
            // std::cout<<"Normal exp buffer: "<<_normal_obs_dynamics.size()<<std::endl;
            std::cout<<"data limit       : "<<_data_limit<<std::endl;

            /** --------------------------------------------------------------------------------
             * Archive executed and expected behaviors/trajectories
             * for novelty computation 
             */ 
            Eigen::VectorXd executed_behavior = this->compute_behavior(obs_new);
            this->_obs_behaviors.push_back(executed_behavior);
            this->_expected_behaviors.push_back(get_expected_behavior());
            _last_novelty = this->compute_novelty(executed_behavior);
            _last_behavior = executed_behavior;

            /** ----------------------------------------------------------------------------------
             * Dump observed state trajectory
             */ 
            std::vector<Eigen::VectorXd> states = _robot.get_last_states();           
            std::ofstream obs_state_trajectory;
            obs_state_trajectory.open("state_traject_actual_" + std::to_string(ep) + ".dat");
            for (size_t i=0; i< states.size(); i++)
            {
                for(size_t j=0; j < Params::multidex::model_pred_dim(); j++)
                {
                    obs_state_trajectory<<states[i](j)<<",";
                }
                obs_state_trajectory<<std::endl;
            }
            obs_state_trajectory.close();
            _policy_set = false;
        }


        /** Returns minimum distance stored expected behaviors */    
        float compute_novelty(const Eigen::VectorXd& behavior) const
        {

            float min_dist = std::numeric_limits<double>::max();
            for(int i=0; i< this->_expected_behaviors.size(); i++) {
                float dist = 0;   
                dist = (behavior - this->_expected_behaviors[i]).squaredNorm();
                if(dist<min_dist)
                {
                    min_dist = dist;
                }
            }
            return min_dist;
        }

        /** Accumulates all the rewards for the given observation/trajectory */
        float get_reward(const obsDynamics_t& obs) const
        {
            double r = 0;
            for(int i = 0; i< obs.size(); i++)
            {
                Eigen::VectorXd from_state = std::get<3>(obs[i]);
                Eigen::VectorXd action = std::get<1>(obs[i]);
                Eigen::VectorXd to_state = std::get<3>(obs[i]) + std::get<2>(obs[i]);
                r += _reward_func(from_state, action, to_state);
            }
            return r;
        } 

        /** Returns novelty, reward and -variance score for the policy 
         * NOTE: This function can be called in parallel 
         * SO use mutex.lock() for updating class member variables
         */ 
        std::vector<float> _fitness(const std::vector<float>& indivData) 
        {            
            _mutex.lock();
            _opt_iters++;
            _mutex.unlock();
            std::vector<double>  indiv (indivData.begin(), indivData.end());
            Eigen::VectorXd params = Eigen::VectorXd::Map(indiv.data(), indiv.size()); //TODO: Check if data is transferring correctly
            Policy policy;
            policy.set_params(params); 
            obsDynamics_t obs;
            std::vector<double> model_variance;

            obs = _robot.predict_policy(policy, this->_model, Params::multidex::T(),model_variance);

            Eigen::VectorXd behavior = this->compute_behavior(obs);
            float n = this->compute_novelty(behavior);
            float r = this->get_reward(obs);
            std::vector<float> fitness(3);

            fitness[0] = n;
            fitness[1] = -std::accumulate(model_variance.begin(), model_variance.end(), 0.0) / model_variance.size();            
            fitness[2] = r;
            return fitness;
        }

        /** Just for debugging purpose */
        void execute_model()
        {
            Policy policy;
            policy.set_params(_policy.params()); 
            obsDynamics_t obs;
            obs = this->_robot.predict_policy(policy, this->_model, Params::multidex::T());
            Eigen::VectorXd behavior = this->compute_behavior(obs);
            std::cout<<"Expected Behavior: "<<behavior.transpose()<<std::endl;
        }

        /** Get expected behavior for the currently set policy 
         *  Should be called immediately after execution to 
         *  to compute the expected behavior/trajectory by the\
         *  current model.
        */
        Eigen::VectorXd get_expected_behavior()
        {
            Policy policy;
            policy.set_params(_policy.params()); 
            obsDynamics_t obs;
            obs = this->_robot.predict_policy(policy, this->_model, Params::multidex::T());
            return this->compute_behavior(obs);
        }

        /** Limit data for dynamics learning 
         *  simply keep the recent _data_limit number of points 
         */
        obsDynamics_t limit_obsDynamics(const obsDynamics_t& obsDynamics)
        {        
            if (_data_limit > obsDynamics.size())
                return obsDynamics;

            obsDynamics_t limited_obsDynamics(obsDynamics.end()-_data_limit, obsDynamics.end());
            return limited_obsDynamics;
        }

        /** Learn model using _limited_obs_dynamics U _best_obs_dynamics*/
        void learn_dynamics_model()
        {
            std::cout<<"Learning dynamics model..."<<std::endl;
            
            // _limited_obs_dynamics = limit_obsDynamics(_obs_dynamics);
            _limited_obs_dynamics = limit_obsDynamics(_normal_obs_dynamics);
            _limited_best_dynamics = limit_bestObsDynamics( size_t(Params::multidex::model_data_limit()/_steps) , _best_obs_dynamics);             
           
            /** Combine limited obs dynamics and best obs dynamics for model learning **/
            _limited_obs_dynamics.insert(this->_limited_obs_dynamics.end(), _limited_best_dynamics.begin(), _limited_best_dynamics.end()); 
            _model.learn(_limited_obs_dynamics);
            _limited_obs_dynamics.clear();
            _limited_best_dynamics.clear();
        }    

        Eigen::VectorXd compute_behavior(const obsDynamics_t& obs_new)
        {
            int samples = Params::multidex::trajectorySamples();
            Eigen::VectorXd behavior(samples*Params::multidex::model_pred_dim());
            int step_size = int(obs_new.size()/samples);
            
            for(int i=1; i<= samples; i++){
                Eigen::VectorXd stateSample = std::get<2>(obs_new[i*step_size - 1]) + std::get<3>(obs_new[i*step_size - 1]); //NOTE: mu+init //TODO: normalize using state limits (joint limits)
                behavior.segment((i-1)*Params::multidex::model_pred_dim(),Params::multidex::model_pred_dim()) = stateSample.cwiseQuotient(_max_predictions);
            }
            return behavior;
        }

        void save_behavior(const Eigen::VectorXd& behavior)
        {
            _obs_behaviors.push_back(behavior);
        }

        /** Should be called after every dynamics model learning command 
         *  Update expected behaviors using the new dynamics model
        */
        void recompute_behaviors()
        {
            std::cout<<"Recomputing expected behaviors.."<<std::endl;
            Policy policy;
            _expected_behaviors.clear();
            for (size_t i = 0; i < _executed_policies.size(); i++)
            {          
                policy.set_params(_executed_policies[i]); 
                obsDynamics_t obs;
                obs = this->_robot.predict_policy(policy, this->_model, Params::multidex::T());
                _expected_behaviors.push_back(this->compute_behavior(obs));
            }
        }

        obsDynamics_t limit_bestObsDynamics(const size_t& episodes, const obsDynamics_t& data)
        {
            if(episodes >= data.size()/_steps)
                return data;
            std::vector<double> top_rew_indices;
            std::vector<double> temp = _best_accumulated_rewards;
            double best = 0;
            size_t best_index;
            obsDynamics_t best_obs;

            for(size_t i=0; i < episodes; i++)
            {    
                for(size_t j=0; j<temp.size(); j++)
                {
                    if(temp[j] > best)
                    {
                        best_index = j;
                        best = temp[j];
                    }
                }
                top_rew_indices.push_back(best_index);
                temp[best_index] = 0;
                best = 0;
            }

            for(size_t i=0; i < top_rew_indices.size(); i++)
            {
                std::cout<<"Best observed rewards: "<<top_rew_indices[i]<<") "<<_best_accumulated_rewards[top_rew_indices[i]]<<std::endl;
                best_obs.insert(best_obs.end(), data.begin() + top_rew_indices[i] * _steps, data.begin() + top_rew_indices[i] * _steps + _steps);
                std::cout<<best_obs.size()<<std::endl;
            }
            return best_obs;
        }
    };
}
#endif
