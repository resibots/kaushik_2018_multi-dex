/** Definitions */
#undef NO_PARALLEL //Make sure sferes2 use parallel processing
/** GP model 
 * and LIMBO unilities
 */

#include<limbo/opt/parallel_repeater.hpp>
#include<limbo/opt/rprop.hpp>
#include<limbo/model/gp.hpp>
#include<limbo/kernel/kernel.hpp>
#include<limbo/tools/macros.hpp>
#include<limbo/tools/math.hpp>

#include<multidex/gp_model.hpp>
#include<multidex/model/multi_gp.hpp>

#include<multidex/model/gp/kernel_lf_opt.hpp>
#include<multidex/model/multi_gp/parallel_lf_opt.hpp>

/** multi-DEX headers */ 
#include<multidex/multidex.hpp>
#include <multidex/policy/nn_policy.hpp>

/** Simulator includes */ 
#include <robot_dart/position_control.hpp>
#include <robot_dart/robot_dart_simu.hpp>
#ifdef GRAPHIC
#include <robot_dart/graphics.hpp>
#endif
#include <multidex/system/dart_system.hpp>

/** Multi-objective optimizer headers
 * from sferes2 library
 */ 
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/ea/nsga2.hpp>
#include <sferes/stat/pareto_front.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/run.hpp>
#ifndef NO_PARALLEL
#include <sferes/eval/parallel.hpp>
#else
#include <sferes/eval/eval.hpp>
#endif

/** Other unilities */ 
#include <chrono>
#include <boost/program_options.hpp>

/** Define namespaces */
using namespace sferes;
using namespace sferes::ea;
using namespace sferes::gen::evo_float;


struct Params{    
    struct multidex{
        BO_PARAM(size_t, action_dim, 2); //5 velocity controls
        BO_PARAM(size_t, model_input_dim, 3); //5 positions
        BO_PARAM(size_t, model_pred_dim, 3); // 5 positions
        BO_PARAM(double, dt, 0.1);
        BO_PARAM(double, T, 4.0);
        BO_PARAM(int, model_data_limit, 250);
        BO_PARAM(int, iteration, 300);
        BO_PARAM(double, define_good_rew, 0.0); //Any rew greater than this is assumed good

        BO_DYN_PARAM(double, epsilon); //probability of exploration than selecting best one   
        BO_PARAM_ARRAY(double, limits, M_PI/3, 3*M_PI/4, 0.5); //TODO: model input absolute max
        BO_PARAM_ARRAY(double, limits_low, -M_PI/6, -M_PI/4, 0); //Model input low
        BO_PARAM_ARRAY(double, model_pred_limits, M_PI/3, 3*M_PI/4, 0.5); //NOTE:: For behavior normalization and constraining pred
        BO_PARAM_ARRAY(double, model_pred_limits_low, -M_PI/6, -M_PI/4, 0); //NOTE:: For behavior normalization and constraining pred
        BO_PARAM(size_t, trajectorySamples, 10); // Must be less than or equal to T/dt
        BO_PARAM(double, custom_init_pop_percent, 0.30); //Percentage of population to be inserted from last observed and found.
        BO_PARAM(double, behavior_buffer_size, 10); //Only keep N most novel behaviors for novelty computation

        struct behavior{
            BO_PARAM(double,dim,5); //NOTE:
        };
    }; 

/** Simulator parameters */
    struct dart_system {
        BO_PARAM(double, sim_step, 0.001);
    };

    struct dart_policy_control {
        BO_PARAM(dart::dynamics::Joint::ActuatorType, joint_type, dart::dynamics::Joint::SERVO);
    };

#ifdef GRAPHIC
    struct graphics : robot_dart::defaults::graphics {
    };
#endif

/** GP model parameters */
    struct gp_model {
        BO_PARAM(double, noise, 0.01);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, gp_model::noise());
        BO_PARAM(bool, optimize_noise, true);
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };
    
    /** GP Hyper parameter optimizer parameters */
    struct opt_rprop : public limbo::defaults::opt_rprop {
        BO_PARAM(int, iterations, 300);
        BO_PARAM(double, eps_stop, 1e-4);
    };

    struct opt_parallelrepeater : public limbo::defaults::opt_parallelrepeater {
        BO_PARAM(int, repeats, 3);
    };

/** Sferes2 Parameters for NSGA-II */
    struct evo_float {
        SFERES_CONST float cross_rate = 0.5f;
        SFERES_CONST float mutation_rate = 0.1f;
        SFERES_CONST float eta_m = 15.0f;
        SFERES_CONST float eta_c = 10.0f;
        SFERES_CONST mutation_t mutation_type = polynomial;
        SFERES_CONST cross_over_t cross_over_type = sbx;
      };

    struct pop {
        SFERES_CONST unsigned size = 200; //NOTE: Must be multiple of 4
        SFERES_CONST unsigned nb_gen = 500;
        SFERES_CONST int dump_period = -1;
        SFERES_CONST int initial_aleat = 1;
    };

    struct parameters {
        SFERES_CONST float min = -3.0f;
        SFERES_CONST float max = 3.0f;
    };
};


/** Policy Parameters */
struct policyParams {
    struct nn_policy {
        BO_PARAM(size_t, state_dim, Params::multidex::model_input_dim());
        BO_PARAM(size_t, action_dim, Params::multidex::action_dim());
        BO_PARAM_ARRAY(double, max_u, 1.0, 1.0); //NOTE Policy output limits
        BO_PARAM_ARRAY(double, limits, M_PI/3, 3*M_PI/4, 0.5); //NOTE: Policy input limits for normalization        
        BO_PARAM(int, hidden_neurons, 5); //NOTE: Only one hidden layer allowed
        BO_PARAM(double, af, 1.0);
        BO_PARAM(double, paramsBound, 3); //NOTE: Must match sferes parameters upper bound
    };
};

/** Global variables */
namespace global {
    std::shared_ptr<robot_dart::Robot> global_robot, gobal_drawer;
    using policy_t = multidex::policy::NNPolicy<policyParams>;
    Eigen::VectorXd goal(3);
    int argc;
    char** argv;
    int evaluations = 0;
    size_t best_rew_pf_point=0;
    Eigen::VectorXd best_policy_so_far;
    bool hasCollided = false;
}


/** The reward function **/
struct ActualRewardFunction{  
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        if(to_state[2] > 0.001)
        {   
            double drawer_rew = double(to_state[2]);
            // double drawer_rew = std::exp(-(to_state[2]-0.5)*(to_state[2]-0.5));
            double reposition_reward = 0;
            if(to_state[2]> 0.2)
                reposition_reward = std::exp(-(to_state[0]-0)*(to_state[0]-0) - (to_state[1]-0)*(to_state[1]-0));
            return drawer_rew + reposition_reward;                
        }
        else
            return 0.0;
    }
};

struct LearnedRewardFunction{
    double operator()(const Eigen::VectorXd& from_state, const Eigen::VectorXd& action, const Eigen::VectorXd& to_state) const
    {
        //Query reward using object of of the model
        return 0;
    }
};

void init_simu(const std::string& robot_file, const std::string& drawer_file)
{
    std::cout<<"Initializing simulation"<<std::endl;
    global::global_robot = std::make_shared<robot_dart::Robot>(robot_dart::Robot(robot_file, {}, "arm", true));
    global::gobal_drawer = std::make_shared<robot_dart::Robot>(robot_dart::Robot(drawer_file, {}, "drawer", true));

    /** Make the task harder by removing the handle friction 
        so that the arm has to latch properly to open the drawer.
    */
    global::gobal_drawer->skeleton()->getBodyNode("handle_grip")->setFrictionCoeff(0.00);
    global::gobal_drawer->skeleton()->getBodyNode("handle_left")->setFrictionCoeff(0.00);
    global::gobal_drawer->skeleton()->getBodyNode("handle_right")->setFrictionCoeff(0.00);
    global::gobal_drawer->skeleton()->getBodyNode("handle_top_cover")->setFrictionCoeff(0.00);
}

Eigen::VectorXd get_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, const std::shared_ptr<robot_dart::Robot>& drawer)
{
    Eigen::VectorXd pos = robot->skeleton()->getPositions();
    size_t size = pos.size() + 1;
    Eigen::VectorXd state(size);
    state.head(pos.size()) = pos;    
    state.tail(1) = drawer->skeleton()->getPositions();
    return state;
}

struct PolicyBasedController : public multidex::system::BaseDARTPolicyControl<Params, global::policy_t> {
    using base_t = multidex::system::BaseDARTPolicyControl<Params, global::policy_t>;
    using obsDynamics_t = std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>;
    PolicyBasedController() : base_t() {}
    PolicyBasedController(const std::vector<double>& ctrl, base_t::robot_t robot) : base_t(ctrl, robot) {}

    Eigen::VectorXd get_state(const robot_t& robot, bool full) const
    {
        return limbo::tools::make_vector(0);
    }

/** A hack to include the drawer as robot to get the states */
    void set_commands()
    {
        double dt = Params::multidex::dt();

        if (_t == 0.0 || (_t - _prev_time) >= dt) {
            Eigen::VectorXd commands = this->_policy.next(get_robot_state(_robot, drawer));
            Eigen::VectorXd q = get_robot_state(_robot, drawer);
            this->_states.push_back(q);
            this->_coms.push_back(commands);

            assert(_dof == (size_t)commands.size());
            this->_robot->skeleton()->setCommands(commands);
            this->_prev_commands = commands;
            this->_prev_time = _t;
        }
        else
        this->_robot->skeleton()->setCommands(_prev_commands);
    }

public:
    std::shared_ptr<robot_dart::Robot> drawer; //NOTE: To be set before execution
};

struct Manipulator : public multidex::system::DARTSystem<Params, PolicyBasedController> 
{
    using base_t = multidex::system::DARTSystem<Params, PolicyBasedController>;
    using obsDynamics_t = std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>;

    Eigen::VectorXd init_state() const
    {
        Eigen::VectorXd initstate = Eigen::VectorXd::Zero(Params::multidex::model_pred_dim());
        initstate[0] = 0.7;
        initstate[1] = 0;
        return initstate;
    }

    void set_robot_state(const std::shared_ptr<robot_dart::Robot>& robot, const Eigen::VectorXd& state) const
    {
        Eigen::VectorXd robot_state = state.head(Params::multidex::model_input_dim()-1);
        robot->skeleton()->setPositions(robot_state);
    }

    /*To transform the input to the GP and the policy*/
    Eigen::VectorXd transform_state(const Eigen::VectorXd& original_state) const
    {
        return original_state;
    }

    /*Contrain the prediction as long horizon predictions might go beyond permissible state-space for the system*/
    Eigen::VectorXd constrain_prediction(const Eigen::VectorXd& original_state) const
    {
        Eigen::VectorXd constrained = original_state;
        
        for(size_t i=0; i< constrained.size(); i++)
        {
            if(constrained[i] > Params::multidex::model_pred_limits(i))
                constrained[i] = Params::multidex::model_pred_limits(i); 
            if(constrained[i] < Params::multidex::model_pred_limits_low(i))
                constrained[i] = Params::multidex::model_pred_limits_low(i); 
        }

        return constrained;
    }

    std::shared_ptr<robot_dart::Robot> get_robot() const
    {
        std::shared_ptr<robot_dart::Robot> simulated_robot = global::global_robot->clone();
        simulated_robot->fix_to_world();
        simulated_robot->set_position_enforced(true);
        return simulated_robot;
    }

    void add_extra_to_simu(base_t::robot_simu_t& simu) const
    {
#ifdef GRAPHIC
        simu.graphics()->fixed_camera(Eigen::Vector3d(2.0, 2.0, 2.0), Eigen::Vector3d(-0.5, 0.0, 0.4));
#endif
    }

    template <typename Policy, typename Reward>
    obsDynamics_t execute(const Policy& policy, const Reward& world, double T, std::vector<double>& R, bool& collision)
    {
        assert(Params::dart_system::sim_step() < Params::multidex::dt());
        obsDynamics_t res;
        Eigen::VectorXd pp = policy.params();
        std::vector<double> params(pp.size());
        Eigen::VectorXd::Map(params.data(), pp.size()) = pp;
        std::shared_ptr<robot_dart::Robot> simulated_robot = this->get_robot();
        R = std::vector<double>();
        robot_simu_t simu(params, simulated_robot);
        // simulation step different from sampling rate -- we need a stable simulation
        simu.set_step(Params::dart_system::sim_step());
        Eigen::VectorXd init_diff = this->init_state();
        this->set_robot_state(simulated_robot, init_diff);
        
        this->add_extra_to_simu(simu);

        auto c_drawer = global::gobal_drawer->clone();
        simu.controller().drawer = c_drawer;


        Eigen::Vector6d pose = Eigen::VectorXd::Zero(6);
        pose.tail(3) = Eigen::Vector3d(-1.15, 0, 0.4);
        pose.head(3) = Eigen::Vector3d(0, 0, 0);
        simu.add_skeleton(c_drawer->skeleton(), pose, "fixed", "drawer");

        Eigen::Vector3d gravity;
        gravity << 0.0, 0.0, 0.0;
        simu.world()->setGravity(gravity);
        
        // simu.set_step(0.01);
        // simu.add_floor();

        simu.run(T + Params::multidex::dt());
        
        std::vector<Eigen::VectorXd> states = simu.controller().get_states();
        _last_states = states;
        std::vector<Eigen::VectorXd> commands = simu.controller().get_commands();
        _last_commands = commands;

        for (size_t j = 0; j < states.size() - 1; j++) {
            Eigen::VectorXd init = states[j];
            Eigen::VectorXd init_full = this->transform_state(init);
            Eigen::VectorXd u = commands[j];
            Eigen::VectorXd final = states[j + 1];
            double r = world(init, u, final);
            R.push_back(r);
            res.push_back(std::make_tuple(init_full, u, final - init, init));
        }
        auto w = simu.world();
        //need to bake() inside robot dart
        auto recorder= w->getRecording();
        int num_contacts = 0;
        for(int i=0; i < recorder->getNumFrames(); i++)
            num_contacts += recorder->getNumContacts(i);

        std::cout<<"Total Contacts:  "<<num_contacts<<std::endl;
        auto collisionRes = w->getLastCollisionResult();
        std::cout<<"Last collision Contacts:  "<<collisionRes.getNumContacts()<<std::endl;
        // if(collisionRes.getNumContacts()>0)
        if(num_contacts>0)
            global::hasCollided = true;
        else
            global::hasCollided = false;
        collision = global::hasCollided;
        return res;
    }

    template <typename Policy, typename Model>
    obsDynamics_t predict_policy(const Policy& policy, const Model& model, double T, std::vector<double>& variance) const
    {
        double dt = Params::multidex::dt();
        obsDynamics_t observations;
        Eigen::VectorXd init_raw = this->init_state();
        
        for (double t = 0.0; t <= T; t += dt) {
            
            Eigen::VectorXd init = this->transform_state(init_raw);
            Eigen::VectorXd query_vec(Params::multidex::model_input_dim() + Params::multidex::action_dim());
            Eigen::VectorXd u = policy.next(init);
            query_vec.head(Params::multidex::model_input_dim()) = init;
            query_vec.tail(Params::multidex::action_dim()) = u;

            Eigen::VectorXd mu;
            Eigen::VectorXd sigma;
            std::tie(mu, sigma) = model.predictm(query_vec);

            sigma = sigma.array().sqrt();
            Eigen::VectorXd final = init_raw + mu;
            final = this->constrain_prediction(final);
            
            std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> transition;
            std::get<0>(transition) = init; //Transformed initial state
            std::get<1>(transition) = u;    //action
            std::get<2>(transition) = final - init_raw; //Predicted state  
            std::get<3>(transition) = init_raw;  //Raw initial state.
            observations.push_back(transition);
            
            init_raw = final;
            variance.push_back(sigma.sum());
        }
        return observations;
    }
    
    template <typename Policy, typename Model>
    obsDynamics_t predict_policy(const Policy& policy, const Model& model, double T) const
    {
        std::vector<double> temp;
        return predict_policy(policy, model, T, temp);
    }
};

// -------------------------------------------------
using kernel_t = limbo::kernel::SquaredExpARD<Params>;
using mean_t = limbo::mean::Constant<Params>;
using rewardModel_t = ActualRewardFunction;
using GP_t = multidex::model::MultiGP<Params, limbo::model::GP, kernel_t, mean_t, multidex::model::multi_gp::MultiGPParallelLFOpt<Params, multidex::model::gp::KernelLFOpt<Params>>>;
using Dynamics_model_t = multidex::GPModel<Params, GP_t>;
multidex::Multidex<Params, global::policy_t, rewardModel_t, Manipulator, Dynamics_model_t> mdx;
// -------------------------------------------------


SFERES_FITNESS(FitZDT2, sferes::fit::Fitness) {
public:
  FitZDT2()  {}
  template<typename Indiv>
  void eval(Indiv& ind) {
    this->_objs.resize(3);
    std::vector<float> fitness;
    fitness = mdx._fitness(ind.data());
    float f1 = fitness[0]; //n 
    float f2 = fitness[1]; //d
    float f3 = fitness[2]; //r
    this->_objs[0] = f1;
    this->_objs[1] = f2;
    this->_objs[2] = f3;
    global::evaluations++;

#if defined(CONSOLE)
    if(global::evaluations%50 == 0)
        std::cout<<"\r"<<double(global::evaluations/double(Params::pop::nb_gen * Params::pop::size))*100<<"%     "<<std::flush;
#endif
  }
};

typedef eval::Parallel<Params> eval_t;
typedef gen::EvoFloat<32, Params> gen_t;
typedef phen::Parameters<gen_t, FitZDT2<Params>, Params> phen_t;
typedef boost::fusion::vector<stat::ParetoFront<phen_t, Params>>  stat_t;
typedef modif::Dummy<> modifier_t;

namespace global{
    size_t pareto_front_size = 0;
    std::vector<boost::shared_ptr<phen_t>> pareto_front;
}

template<typename Phen, typename Eval, typename Stat, typename FitModifier, typename Params, typename Exact = stc::Itself>
class NSGA2_Custominit : public ea::Nsga2 <Phen, Eval, Stat, FitModifier, Params, typename stc::FindExact<NSGA2_Custominit<Phen, Eval, Stat, FitModifier, Params, Exact>, Exact>::ret>
{
    
public:
      typedef boost::shared_ptr<crowd::Indiv<Phen> > indiv_t;
      typedef typename std::vector<indiv_t> pop_t;
      typedef typename pop_t::iterator it_t;
      typedef typename std::vector<std::vector<indiv_t> > front_t;

    void random_pop() {
        this->_parent_pop.resize(Params::pop::size);
        assert(Params::pop::size % 4 == 0);

        size_t insert_pop_size = 0;
        if(size_t(Params::pop::size * Params::multidex::custom_init_pop_percent()) > global::pareto_front_size)
            insert_pop_size = global::pareto_front_size;
        else
            insert_pop_size = size_t(Params::pop::size * Params::multidex::custom_init_pop_percent());

        pop_t init_pop( Params::pop::size );
        parallel::p_for(parallel::range_t(0, init_pop.size()),
                        ea::random<crowd::Indiv<Phen> >(init_pop));
        std::cout<<"NSGA2 population initializing..."<<std::endl;
        std::cout<<"Max Insertion size : "<<size_t(Params::pop::size * Params::multidex::custom_init_pop_percent())<<std::endl;
        std::cout<<"PF size            : "<<global::pareto_front_size<<std::endl;
        std::cout<<"Insertion size     : "<<insert_pop_size<<std::endl;
        std::cout<<"Total pop size     : "<<Params::pop::size<<std::endl;      
        
        //Replace some initial random individuals with individuals from the 
        //pareto front.
        std::vector<int> indices;
        for (size_t i = 0; i < global::pareto_front_size; i++)
        {
            indices.push_back(i);
        }    
        std::vector<double> bestPolicyObserved(global::best_policy_so_far.size(), 0.0);
        Eigen::VectorXd::Map(bestPolicyObserved.data(), bestPolicyObserved.size()) = global::best_policy_so_far;
        double bound = policyParams::nn_policy::paramsBound();
        srand (time(NULL));
        for(int i = 0; i<insert_pop_size; i++)
        {       
            
            int ind = rand() % indices.size();
            int random_pf_point = indices[ind];
            indices.erase(indices.begin() + ind);

            for(size_t index=0; index < global::pareto_front[0]->data().size(); index++)
            {
                //NOTE: Insert the OBSERVED best so far policy as well as best policy from the last optimization
                if(i==0)
                {
                    //NOTE: Genotype params should be in [0,1]                    
                    // init_pop[i]->gen().data(index, (global::pareto_front[global::best_rew_pf_point]->data()[index] + bound)/(2.* bound));
                    // continue; 
                }
                
                //Insert the best observed policy finally
                if(i == insert_pop_size - 1 && init_pop.size() >= insert_pop_size + 1 && bestPolicyObserved.size() > 0)
                {
                    init_pop[i+1]->gen().data(index, (bestPolicyObserved[index] + bound) / (2.* bound));
                }
                
                //NOTE: Genotype params should be in [0,1]                    
                init_pop[i]->gen().data(index, (global::pareto_front[random_pf_point]->data()[index] + bound) / (2.* bound));
                
            }
        }
        std::cout<<"NSGA2 population initialized"<<std::endl;
        this->_eval_subpop(init_pop);
        this->_apply_modifier(init_pop);
        front_t fronts;
        this->_rank_crowd(init_pop, fronts);
        this->_fill_nondominated_sort(init_pop, this->_parent_pop);
    }
};



typedef NSGA2_Custominit <phen_t, eval_t, stat_t, modifier_t, Params> ea_t;


BO_DECLARE_DYN_PARAM(double, Params::multidex, epsilon);

int main(int argc, char** argv)
{
    /** ----------------------------------------
     * Enable tbb parallel
     */ 
    int threads = tbb::task_scheduler_init::automatic; 
    static tbb::task_scheduler_init init(threads);

    

    /** ----------------------------------------
     * Results dump files
     */ 
    std::ofstream ofs_results, ofs_pareto_front, ofs_behaviors, ofs_optimization_time, ofs_learning_time;
    ofs_optimization_time.open("opt_time.dat");
    ofs_learning_time.open("learning_time.dat");
    ofs_results.open("results.dat");
    ofs_behaviors.open("behaviors.dat");

    /** -----------------------------------------
     * Commandline args
     */ 
    global::argc = argc;
    global::argv = argv;
    
    namespace po = boost::program_options;
    po::options_description desc("Command line arguments");
    // clang-format off
    desc.add_options()("help,h", "Prints this help message")
                      ("epsilon,p", po::value<double>(), "Probability of exploration");

    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);

        
        if (vm.count("epsilon")) {
            double c = vm["epsilon"].as<double>();
            if (c < 0 or c > 1)
            {
                std::cout<<"Epsilon must be a probability"<<std::endl;
                return 0;
            }
            Params::multidex::set_epsilon(c);
        }
        else {
            std::cout<<"Setting default epsilon = 0.5"<<std::endl;
            Params::multidex::set_epsilon(0.5);            
        }
    }
    catch (po::error& e) {
        std::cerr << "[Exception caught while parsing command line arguments]: " << e.what() << std::endl;
        return 1;
    }
   
             
    /** --------------------------------------------------------
     * Load the robot URDF.
     * PS: Separate urdf in cluster with absolute path
     */

    const char* env_p = std::getenv("RESIBOTS_DIR");    
    if (env_p)
    {
        init_simu(std::string(RESPATH) + "/URDF/omnigrasper_hook_for_drawer.urdf", std::string(RESPATH) + "/URDF/drawer_one_sided_handle.urdf"); //if it does not exist, we might be running this on the cluster
    }
    else
    {
        init_simu("/nfs/hal01/rkaushik/shared/URDF/omnigrasper_hook_for_drawer.urdf", "/nfs/hal01/rkaushik/shared/URDF/drawer_one_sided_handle.urdf");
    }

    
    /** --------------------------------------------------------
     * Print important parameters
     */
    std::cout<<std::endl<<"Parameters:"<<std::endl;
    std::cout<<"----------"<<std::endl<<std::endl;

    std::cout<<"model_data_limit             : "<<Params::multidex::model_data_limit()<<std::endl;
    std::cout<<"epsilon                : "<<Params::multidex::epsilon()<<std::endl;
    std::cout<<"Population             : "<<Params::pop::size<<std::endl;
    std::cout<<"Generations            : "<<Params::pop::nb_gen<<std::endl;    

    /** --------------------------------------------------------
     * Initial random episodes
     */ 
    std::chrono::steady_clock::time_point time_start;
    size_t total_episode_count = 0;

    while(total_episode_count < 2)
    {
        std::cout<<std::endl<<"Random trial:"<<std::endl;
        std::cout<<"------------"<<std::endl<<std::endl;
        mdx.execute_random_policy(total_episode_count);
        total_episode_count++;
             
        mdx.learn_dynamics_model();
        global::best_policy_so_far = mdx._best_policy_params;
    }

    /** ------------------------------------------------------
     * Main loop
     */ 
    std::cout<<"Param size = "<<mdx._policy.params().size()<<std::endl;
    for(int iteration=0; iteration < Params::multidex::iteration(); iteration++)
    {    
        ofs_results<<mdx._last_reward<<std::endl;
        ofs_behaviors<<mdx._obs_behaviors.back().transpose()<<std::endl; 
        
        std::cout<<"\n\nEpisode : "<<iteration+1<<std::endl;
        std::cout<<"--------"<<std::endl<<std::endl;
        
        /** ---------------------------------------------------
         * Search Pareto front using NSGA-2
         */ 
        ea_t ea;
        global::evaluations = 0;
        std::cout<<"Searching Pareto front..."<<std::endl;        
        time_start = std::chrono::steady_clock::now();
        mdx._opt_iters = 0;
        ea.run(argv[0]); 
        double optimization_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_start).count();
        ofs_optimization_time << (optimization_ms * 1e-3) << std::endl;

        auto pf = ea.pareto_front();
        global::pareto_front_size = pf.size();
        global::pareto_front = pf;
        std::cout<<std::endl<<"Pareto front size: "<<pf.size()<<std::endl;
        int pf_size = pf.size();
        int pheno_size = pf[0]->size();
        int index_max_reward = 0;
        int index_max_dr = 0;
        int index_max_nov = 0;
        float maxrew = -std::numeric_limits<float>::max();
        float maxdr = 0;
        float maxnov = -std::numeric_limits<float>::max();

        ofs_pareto_front.open("pareto_front_"+ std::to_string(iteration+1) +".dat");
        for(int i=0; i<pf_size;i++)
        {
            ofs_pareto_front << pf[i]->fit().obj(0) << " " << pf[i]->fit().obj(1)<<" " << pf[i]->fit().obj(2)<<std::endl;
            
            //Best reward point with non zero novelty
            if(pf[i]->fit().obj(1) > maxdr && pf[i]->fit().obj(1) < 0.7)
            {
                maxdr = pf[i]->fit().obj(1);
                index_max_dr = i;           
            }

            if(pf[i]->fit().obj(2) > maxrew && pf[i]->fit().obj(1) > -0.08)
            {
                maxrew = pf[i]->fit().obj(2);
                index_max_reward = i;           
            }

            if(pf[i]->fit().obj(0) > maxnov && pf[i]->fit().obj(1) > -0.1)
            {
                maxnov = pf[i]->fit().obj(0);
                index_max_nov = i;           
            }
        }
        ofs_pareto_front.close();
        std::cout<<std::endl;
        global::best_rew_pf_point = index_max_reward;

        srand (time(NULL));
        
        double rand_num = limbo::tools::random_vector(1)[0];   
        std::cout<<"Random = "<<rand_num<<std::endl;
        std::vector<float> temp;
        if(rand_num  <= Params::multidex::epsilon() || pf[index_max_reward]->fit().obj(2) == 0.0)
        {
            int random = rand() % pf_size;
            for(size_t count=0; count < 500; count++)
            {
                int random1 = rand() % pf_size;
                if(pf[random1]->fit().obj(0) > pf[random]->fit().obj(0) && pf[random1]->fit().obj(1) > pf[random]->fit().obj(1))
                    random = random1;
            }
            temp = pf[random]->data();
            std::cout<<"Selected RANDOM N D R : "<< pf[random]->fit().obj(0)<<" "<< pf[random]->fit().obj(1)<<" "<< pf[random]->fit().obj(2)<<std::endl;
        }
        else
        {
            rand_num = limbo::tools::random_vector(1)[0];
            int index;
            if(rand_num < 2)
            {
                index = index_max_reward;
                std::cout<<"Selected BEST R: "<< pf[index]->fit().obj(0)<<" "<< pf[index]->fit().obj(1)<<" "<< pf[index]->fit().obj(2)<<std::endl;
            }
            else
            {
                index = index_max_dr;
                std::cout<<"Selected BEST D: "<< pf[index]->fit().obj(0)<<" "<< pf[index]->fit().obj(1)<<" "<< pf[index]->fit().obj(2)<<std::endl;
            }
            temp = pf[index]->data();
        }
        
        global::best_policy_so_far = mdx._best_policy_params;
        std::vector<double> policyVec(temp.begin(),temp.end());
        Eigen::VectorXd policyNew = Eigen::VectorXd::Map(policyVec.data(), policyVec.size());
        

        mdx._policy.set_params(policyNew);
        mdx._policy_set = true;
        Eigen::write_binary("policy_params_" + std::to_string(iteration+1) + ".bin", policyNew);
        

        time_start = std::chrono::steady_clock::now();
        mdx.execute_model();
        mdx.execute_robot(total_episode_count);
        std::cout<<"Observed N R: "<<mdx._last_novelty<<" "<<mdx._last_reward<<std::endl;
        std::cout<<"Best R: "<<mdx._best_reward<<std::endl;
        ofs_behaviors<<mdx._last_behavior<<" => "<<mdx._last_novelty<<" , "<<mdx._last_reward<<std::endl;

        std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> obs_expected;
        obs_expected = mdx._robot.predict_policy(mdx._policy, mdx._model, Params::multidex::T());
        std::cout<<"Expected with new Rew model: "<<mdx.get_reward(obs_expected)<<std::endl;
        std::ofstream f_expected_obs;
        f_expected_obs.open("state_traject_expected_" + std::to_string(total_episode_count) +".dat");
        
        for(size_t i=0; i < obs_expected.size(); i++)
        {
            Eigen::VectorXd state = std::get<3>(obs_expected[i]) + std::get<2>(obs_expected[i]); 
            for(size_t j=0; j<Params::multidex::model_pred_dim(); j++)
            {
                f_expected_obs<< std::get<3>(obs_expected[i])[j]<<",";
            }
            f_expected_obs<<std::endl;
        }

        f_expected_obs.close();

                //Test using best policy paparms known
        if (mdx._best_policy_params.size() > 0)
        {
            global::policy_t pol;
            pol.set_params(mdx._best_policy_params);
            obs_expected = mdx._robot.predict_policy(pol, mdx._model, Params::multidex::T());
            std::cout<<"Best policy rew (current model, new rew model) : "<<mdx.get_reward(obs_expected)<<std::endl;
        }
         
        //------------------------------------  
        mdx.learn_dynamics_model();

        if (mdx._best_policy_params.size() > 0)
        {
            global::policy_t pol;
            pol.set_params(mdx._best_policy_params);
            obs_expected = mdx._robot.predict_policy(pol, mdx._model, Params::multidex::T());
            std::cout<<"Best policy rew (new model) : "<<mdx.get_reward(obs_expected)<<std::endl;
        }

        obs_expected = mdx._robot.predict_policy(mdx._policy, mdx._model, Params::multidex::T());
        std::cout<<"Expected rew with Rew Dyn model: "<<mdx.get_reward(obs_expected)<<std::endl;
        std::cout<<"Expected Novelty with new Dyn model: "<<mdx.compute_novelty(mdx.get_expected_behavior())<<std::endl;
        mdx.recompute_behaviors();
        std::cout<<"Expected behaviors recomputed"<<std::endl;
        std::cout<<"Expected Novelty with recomputed behavior: "<<mdx.compute_novelty(mdx.get_expected_behavior())<<std::endl;
        double learning_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_start).count();
        ofs_learning_time << (learning_ms * 1e-3) << std::endl;
        
        Eigen::VectorXd expected_behavior = mdx.get_expected_behavior();
        std::cout<<"Policy Params: \n"<<mdx._policy.params().transpose()<<std::endl;
        // std::cout<<"Behavior deviation: "<< (expected_behavior - mdx._last_behavior).array().abs().transpose()<<std::endl;        
        // std::cout<<"Behavior diff: "<< (expected_behavior - mdx._last_behavior).norm()<<std::endl;
        total_episode_count++;
        std::cout<<"Behaviors: "<<mdx._executed_policies.size()<<std::endl;
        
        if(mdx._expected_behaviors.size() > Params::multidex::behavior_buffer_size())
        {
            assert(mdx._expected_behaviors.size() == mdx._executed_policies.size() && "Must match num of expected_behaviors and num of executed  policies");
            size_t min_novelty_index = 0;
            double min_novelty = std::numeric_limits<double>::max();
            std::cout<<"Novelties: ";
            //save expected behaviors first
            std::vector<Eigen::VectorXd> old_expected_behaviors = mdx._expected_behaviors;
            for(size_t i=0; i<old_expected_behaviors.size(); i++)
            {
                //delete the one so we can compute novelty
                mdx._expected_behaviors = old_expected_behaviors;
                mdx._expected_behaviors.erase(mdx._expected_behaviors.begin() + i);
                
                double nov = mdx.compute_novelty(old_expected_behaviors[i]);
                std::cout<<"("<<i<<")"<<nov<<" ";
                if (nov < min_novelty)
                {
                    min_novelty_index = i;
                    min_novelty = nov;
                }
            }
            std::cout<<std::endl<<"Deleted: "<<min_novelty_index<<"th executed policy"<<std::endl;
            mdx._executed_policies.erase(mdx._executed_policies.begin() + min_novelty_index);
            mdx.recompute_behaviors();
        }
        std::cout<<"Executed behavior: "<<mdx._obs_behaviors.back().transpose()<<std::endl;
    }
    ofs_optimization_time.close();
    ofs_learning_time.close();
    ofs_results.close();
    ofs_behaviors.close();
    return 0;
}