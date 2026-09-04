#include "controllers/dwmpc/dwmpc.hpp"
#include <cstdlib>

namespace
{
    std::string configPath()
    {
        const char *override_path = std::getenv("DWMPC_CONFIG_PATH");
        return override_path != nullptr
            ? override_path
            : "/usr/include/dls2/controllers/dwmpc/config/config.yaml";
    }
}

namespace controllers
{
    Dwmpc::Dwmpc() : ocp_(), config(YAML::LoadFile(configPath()))
    {}
    Dwmpc::~Dwmpc()
    { }
    void Dwmpc::setWeight(const std::map<std::string,std::vector<double>> &weight_vec)
    {
        weight_vec_ = weight_vec;
        weights_.position = Eigen::Map<const Eigen::Vector3d>(weight_vec_.at("p").data());
        weights_.orientation = Eigen::Map<const Eigen::Vector3d>(weight_vec_.at("quat").data());
        weights_.joint_position = weight_vec_.at("q").at(0);
        weights_.linear_velocity = Eigen::Map<const Eigen::Vector3d>(weight_vec_.at("dp").data());
        weights_.angular_velocity = Eigen::Map<const Eigen::Vector3d>(weight_vec_.at("omega").data());
        weights_.joint_velocity = weight_vec_.at("dq").at(0);
        weights_.torque = weight_vec_.at("tau").at(0);
        weights_.ground_reaction_force = weight_vec_.at("grf").at(0);
        weights_.foot_stance = Eigen::Map<const Eigen::Vector3d>(weight_vec_.at("foot_stance").data());
        weights_.foot_swing = Eigen::Map<const Eigen::Vector3d>(weight_vec_.at("foot_swing").data());
        weights_.consensus = weight_vec_.at("consensus").at(0);
        weights_.gamma = weight_vec_.at("gamma").at(0);
    }
    void Dwmpc::init()
    {   
        std::cout << "DWMPC initialization begins..." << std::endl;

        N_ = config["N_step"].as<int>();

        dt_ = config["dt"].as<double>();
        
        n_joint_wb_ = config["n_joint_wb"].as<int>();
        
        n_contact_wb_ = config["n_contact_wb"].as<int>();
        
        for(int i{0};i < n_contact_wb_;i++)
        {
            terrain_height_.push_back(0); // terrain height for each leg
            early_contact_.push_back(false);
        }  

        q0_ = config["q0"].as<std::vector<double>>();

        foot0_ = config["foot0"].as<std::vector<double>>();

        do_init_ = true;
       
        parameter config_param{};

        config_param.n_problem = config["n_problem"].as<int>();
        
        config_param.subsystems_name = config["subsystems_name"].as<std::vector<std::string>>();

        for(int subsystem{0};subsystem < config_param.subsystems_name.size();subsystem++)
        {
            config_param.subsystems_map_joint[config_param.subsystems_name[subsystem]] = config["subsystems_map_joint"][config_param.subsystems_name[subsystem]].as<std::vector<int>>();
            config_param.subsystems_map_contact[config_param.subsystems_name[subsystem]] = config["subsystems_map_contact"][config_param.subsystems_name[subsystem]].as<std::vector<int>>();
        }
                
        config_param.N_ =  config["N_step"].as<int>();

        config_param.dt =  config["dt"].as<double>();

        config_param.n_contact_wb = n_contact_wb_;

        config_param.n_contact = n_contact_wb_ / config["n_problem"].as<int>();

        config_param.n_joint_wb = n_joint_wb_;

        config_param.n_joint = n_joint_wb_ / config["n_problem"].as<int>();

        config_param.n_state = config["n_state"].as<int>();

        config_param.n_control = config["n_control"].as<int>();

        config_param.torque_limit = config["torque_limit"].as<std::vector<double>>();
        if (config_param.torque_limit.size() != static_cast<std::size_t>(n_joint_wb_))
        {
            throw std::invalid_argument("torque_limit must contain one positive limit per joint");
        }
        for (double limit : config_param.torque_limit)
        {
            if (limit <= 0.0)
            {
                throw std::invalid_argument("torque_limit values must be positive");
            }
        }

        config_param.friction_coefficient = config["friction_coefficient"].as<double>();
        config_param.normal_force_min = config["normal_force_min"].as<double>();
        config_param.normal_force_max = config["normal_force_max"].as<double>();
        config_param.no_slip_velocity = config["no_slip_velocity"].as<double>();
        if (config_param.friction_coefficient <= 0.0
            || config_param.normal_force_min < 0.0
            || config_param.normal_force_max < config_param.normal_force_min
            || config_param.no_slip_velocity < 0.0)
        {
            throw std::invalid_argument("invalid contact constraint parameters");
        }

        ocp_.init(config_param);

        // set the desired to default values

        desired_["robot_height"] = config["robot_height"].as<std::vector<double>>();
        
        desired_["step_height"] = config["step_height"].as<std::vector<double>>();

        desired_["rpy"] = std::vector<double>(3,0);
        
        desired_["dp"] = config["dp"].as<std::vector<double>>();
        
        desired_["omega"] = config["omega"].as<std::vector<double>>();

        weight_vec_["p"] = config["weight_p"].as<std::vector<double>>();
        
        weight_vec_["quat"] = config["weight_quat"].as<std::vector<double>>();
        
        weight_vec_["q"] = config["weight_q"].as<std::vector<double>>();
        
        weight_vec_["dp"] = config["weight_dp"].as<std::vector<double>>();
        
        weight_vec_["omega"] = config["weight_omega"].as<std::vector<double>>();
        
        weight_vec_["dq"] = config["weight_dq"].as<std::vector<double>>();
        
        weight_vec_["tau"] = config["weight_tau"].as<std::vector<double>>();
        
        weight_vec_["grf"] = config["weight_grf"].as<std::vector<double>>();
        
        weight_vec_["foot_stance"] = config["weight_foot_stance"].as<std::vector<double>>();
        
        weight_vec_["foot_swing"] = config["weight_foot_swing"].as<std::vector<double>>();
        
        weight_vec_["consensus"] = config["weight_consensus"].as<std::vector<double>>();

        weight_vec_["gamma"] = config["gamma"].as<std::vector<double>>();
        setWeight(weight_vec_);

        timer_.setDelta(config["delta"].as<std::vector<double>>());
        timer_.setParam(config["duty_factor"].as<double>(),config["step_freq"].as<double>());
        timer_.set({0,0,0,0},{true,true,true,true}); 

        std::cout << "Dwmpc initialized!!!" << std::endl;

    }
    MpcResult Dwmpc::run(const Eigen::Ref<const Eigen::VectorXd> &p,
                         const Eigen::Ref<Eigen::Vector4d> &quat_xyzw,
                         const Eigen::Ref<const Eigen::VectorXd> &q,
                         const Eigen::Ref<const Eigen::VectorXd> &linear_velocity,
                         const Eigen::Ref<const Eigen::VectorXd> &angular_velocity,
                         const Eigen::Ref<const Eigen::VectorXd> &joint_velocity,
                         const double &loop_dt,
                         const Eigen::Ref<const Eigen::Vector4d> &measured_contact,
                         const Eigen::Ref<const Eigen::MatrixXd> &foot_position,
                         const Eigen::Ref<const Eigen::VectorXd> &desired_linear_speed,
                         const Eigen::Ref<const Eigen::VectorXd> &desired_angular_speed)
    {
        if (p.size() != 3 || q.size() != kNumJoints || linear_velocity.size() != 3
            || angular_velocity.size() != 3 || joint_velocity.size() != kNumJoints
            || foot_position.rows() != kNumContacts || foot_position.cols() != 3)
        {
            throw std::invalid_argument("invalid Dwmpc::run input dimensions");
        }

        const Eigen::Quaterniond orientation(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]);
        RobotState state;
        state.position = p;
        state.rpy = quatToRPY(orientation);
        state.joint_position = q;
        state.linear_velocity = linear_velocity;
        state.angular_velocity = angular_velocity;
        state.joint_velocity = joint_velocity;
        state.measured_contact = measured_contact;

        const std::vector<double> contact_vector = timer_.run(loop_dt);
        state.commanded_contact = Eigen::Map<const ContactVector>(contact_vector.data());
        time_ += loop_dt;

        std::vector<double> timer_phase;
        std::vector<bool> timer_initialized;
        timer_.get(timer_phase, timer_initialized);

        const Eigen::MatrixXd feet_world = foot_position.transpose();
        updateTerrainHeight(state.commanded_contact, feet_world);
        for (int leg = 0; leg < kNumContacts; ++leg)
        {
            state.foot_position.segment<3>(3*leg) = feet_world.col(leg);
        }

        desired_["rpy"] = {0, 0, 0};
        desired_["dp"][0] = desired_linear_speed[0];
        desired_["dp"][1] = desired_linear_speed[1];
        desired_["dp"][2] = desired_linear_speed[2];
        desired_["omega"][0] = desired_angular_speed[0];
        desired_["omega"][1] = desired_angular_speed[1];
        desired_["omega"][2] = desired_angular_speed[2];

        ReferenceTrajectory reference;
        setDesiredAndParameter(state.commanded_contact, feet_world, state, reference);
        if (do_sine_wave_)
        {
            sineWave(reference);
        }

        const bool solve_success = ocp_.solve(do_init_, state, reference, weights_);
        timer_.set(timer_phase, timer_initialized);
        return ocp_.getResult(state.commanded_contact, solve_success);
    }

    void Dwmpc::updateTerrainHeight(const ContactVector &contact0,
                                    const Eigen::MatrixXd &foot_op)
    {
        for(int leg{0};leg < n_contact_wb_;leg++)
        {
            if(contact0[leg] > 0)
            {
                // terrain_height_[leg] = -0.33;
                terrain_height_[leg] = foot_op(2,leg);
            }
        }
    }
    double Dwmpc::proprioHeight(double desired_height)
    {   
        double height{0};
        for(int leg{0};leg < n_contact_wb_;leg++)
        {
            height += terrain_height_[leg]/n_contact_wb_;

        }
        return height + desired_height;
    }
    void Dwmpc::setGaitParam(const double duty_factor, const double step_freq, const std::vector<double> &delta)
    {
        timer_.setDelta(delta);
        timer_.setParam(duty_factor,step_freq);
    }
    void Dwmpc::updateTimer(const std::vector<double> &t, const std::vector<bool> &init)
    {
        timer_.set(t,init);
    }
    void Dwmpc::setDesiredAndParameter(const ContactVector &contact0,
                                       const Eigen::MatrixXd &foot_op,
                                       const RobotState &state,
                                       ReferenceTrajectory &reference)
    {
        std::vector<bezier_curves_t> swing_curves(kNumContacts);
        std::vector<bool> early_contact = early_contact_;
        std::vector<bool> fix_swing(kNumContacts);
        double num_contacts = contact0.sum();

        const Eigen::Vector3d desired_rpy = Eigen::Map<const Eigen::Vector3d>(desired_.at("rpy").data());
        const Eigen::Vector3d desired_linear_velocity = Eigen::Map<const Eigen::Vector3d>(desired_.at("dp").data());
        const Eigen::Vector3d desired_angular_velocity = Eigen::Map<const Eigen::Vector3d>(desired_.at("omega").data());
        const JointVector home_joint_position = Eigen::Map<const JointVector>(q0_.data());

        Eigen::Vector3d position = state.position;
        position[2] = proprioHeight(desired_.at("robot_height")[0]);
        FootVector feet;
        for (int leg = 0; leg < kNumContacts; ++leg)
        {
            feet.segment<3>(3*leg) = foot_op.col(leg);
            fix_swing[leg] = contact0[leg] < 1;
        }

        JointVector torque = JointVector::Zero();
        FootVector grf = FootVector::Zero();
        for (int leg = 0; leg < kNumContacts; ++leg)
        {
            grf[3*leg + 2] = 220/std::max(1.0, num_contacts)*contact0[leg];
        }

        reference.position.push_back(position);
        reference.rpy.push_back(desired_rpy);
        reference.joint_position.push_back(home_joint_position);
        reference.linear_velocity.push_back(desired_linear_velocity);
        reference.angular_velocity.push_back(desired_angular_velocity);
        reference.joint_velocity.push_back(JointVector::Zero());
        reference.torque.push_back(torque);
        reference.ground_reaction_force.push_back(grf);
        reference.contact_schedule.push_back(contact0);

        std::vector<double> leg_phase;
        std::vector<bool> timer_initialized;
        timer_.get(leg_phase, timer_initialized);
        for (int leg = 0; leg < kNumContacts; ++leg)
        {
            if (contact0[leg] > 0 && early_contact_[leg])
            {
                early_contact[leg] = false;
                early_contact_[leg] = false;
            }
            if (!early_contact_[leg]
                && contact0[leg] < 1
                && state.measured_contact[leg] > 0
                && std::min((leg_phase[leg]-timer_.duty_factor)/(1-timer_.duty_factor), 1.0) > 0.6)
            {
                early_contact[leg] = true;
                early_contact_[leg] = true;
            }
        }

        for (int k = 0; k < N_+1; ++k)
        {
            const std::vector<double> next_contact_vector = timer_.run(dt_);
            const ContactVector next_contact = Eigen::Map<const ContactVector>(next_contact_vector.data());
            reference.contact_schedule.push_back(next_contact);
            num_contacts = next_contact.sum();

            position += desired_linear_velocity*dt_;
            reference.position.push_back(position);
            reference.rpy.push_back(desired_rpy);
            reference.joint_position.push_back(home_joint_position);
            reference.linear_velocity.push_back(desired_linear_velocity);
            reference.angular_velocity.push_back(desired_angular_velocity);
            reference.joint_velocity.push_back(JointVector::Zero());

            if (k < N_)
            {
                for (int leg = 0; leg < kNumContacts; ++leg)
                {
                    if (reference.contact_schedule[k+1][leg] == 0
                        && reference.contact_schedule[k][leg] != 0)
                    {
                        if (k == 0)
                        {
                            liftoff_pos_[leg] = feet.segment<3>(3*leg) - position;
                        }

                        const double yaw = state.rpy[2];
                        Eigen::Vector3d foothold(
                            std::cos(yaw)*foot0_[3*leg] - std::sin(yaw)*foot0_[3*leg+1],
                            std::cos(yaw)*foot0_[3*leg+1] + std::sin(yaw)*foot0_[3*leg],
                            terrain_height_[leg] - position[2]);
                        foothold[0] += 0.5*desired_linear_velocity[0];
                        foothold[1] += 0.5*desired_linear_velocity[1];
                        foothold[0] += std::sqrt(desired_.at("robot_height")[0]/9.81)
                                     *(state.linear_velocity[0] - desired_linear_velocity[0]);
                        foothold[1] += std::sqrt(desired_.at("robot_height")[0]/9.81)
                                     *(state.linear_velocity[1] - desired_linear_velocity[1]);

                        constexpr double scaling_factor = 0.7105;
                        constexpr double delta_x = 0.10;
                        const double step_height = desired_.at("step_height")[0];
                        const Eigen::Vector3d relative_foot = feet.segment<3>(3*leg) - position;
                        std::vector<Eigen::Vector3d> control_points{
                            relative_foot,
                            relative_foot + Eigen::Vector3d(-delta_x/scaling_factor, 0, step_height/scaling_factor),
                            (relative_foot + foothold)/2 + Eigen::Vector3d(-delta_x/(2*scaling_factor), 0, step_height/scaling_factor),
                            foothold + Eigen::Vector3d(0, 0, step_height/scaling_factor),
                            foothold};
                        bezier_curves_t::curve_constraints_t constraints;
                        constraints.end_vel = Eigen::Vector3d::Zero();
                        bezier_curves_t curve(control_points.begin(), control_points.end(), constraints,
                                              (leg_phase[leg]-timer_.duty_factor)/(1-timer_.duty_factor), 1);
                        swing_curves[leg] = curve;
                        if (k <= 1)
                        {
                            bcs_[leg] = curve;
                        }
                    }

                    if (reference.contact_schedule[k][leg] == 0)
                    {
                        double phase = leg_phase[leg];
                        if (phase < timer_.duty_factor)
                        {
                            phase = 0.99;
                        }
                        const double swing_phase = std::min(
                            (phase-timer_.duty_factor)/(1-timer_.duty_factor), 1.0);
                        if (fix_swing[leg])
                        {
                            if (early_contact[leg])
                            {
                                continue;
                            }
                            const double yaw = state.rpy[2];
                            Eigen::Vector3d foothold(
                                std::cos(yaw)*foot0_[3*leg] - std::sin(yaw)*foot0_[3*leg+1],
                                std::cos(yaw)*foot0_[3*leg+1] + std::sin(yaw)*foot0_[3*leg],
                                terrain_height_[leg] - position[2]);
                            foothold[0] += 0.5*desired_linear_velocity[0];
                            foothold[1] += 0.5*desired_linear_velocity[1];
                            foothold[0] += std::sqrt(desired_.at("robot_height")[0]/9.81)
                                         *(state.linear_velocity[0] - desired_linear_velocity[0]);
                            foothold[1] += std::sqrt(desired_.at("robot_height")[0]/9.81)
                                         *(state.linear_velocity[1] - desired_linear_velocity[1]);

                            constexpr double scaling_factor = 0.7105;
                            constexpr double delta_x = 0.10;
                            const double step_height = desired_.at("step_height")[0];
                            std::vector<Eigen::Vector3d> control_points{
                                liftoff_pos_[leg],
                                liftoff_pos_[leg] + Eigen::Vector3d(-delta_x/scaling_factor, 0, step_height/scaling_factor),
                                (liftoff_pos_[leg] + foothold)/2 + Eigen::Vector3d(-delta_x/(2*scaling_factor), 0, step_height/scaling_factor),
                                foothold + Eigen::Vector3d(0, 0, step_height/scaling_factor),
                                foothold};
                            bezier_curves_t::curve_constraints_t constraints;
                            constraints.end_vel = Eigen::Vector3d::Zero();
                            bezier_curves_t curve(control_points.begin(), control_points.end(), constraints, 0, 1);
                            feet.segment<3>(3*leg) = curve(swing_phase) + position;
                        }
                        else
                        {
                            feet.segment<3>(3*leg) = swing_curves[leg](swing_phase) + position;
                        }
                    }
                    else
                    {
                        early_contact[leg] = false;
                        fix_swing[leg] = false;
                    }
                }

                reference.torque.push_back(torque);
                for (int leg = 0; leg < kNumContacts; ++leg)
                {
                    grf.segment<3>(3*leg).setZero();
                    grf[3*leg+2] = 220/std::max(1.0, num_contacts)*next_contact[leg];
                }
                reference.ground_reaction_force.push_back(grf);
            }
            reference.foot_position.push_back(feet);
            timer_.get(leg_phase, timer_initialized);
        }
    }

    void Dwmpc::sineWave(ReferenceTrajectory &reference)
    {
        double time{time_};
        for(int k{0};k<N_+1;k++)
        {    
            time += dt_;
            reference.position[k][2] += amplitude_*sin(time * 2*M_PI*frequency_);
            reference.linear_velocity[k][2] = 2*M_PI*frequency_ * amplitude_*cos(time * 2*M_PI*frequency_);
        }
    }
    void Dwmpc::setSineParam(double frequency, double amplitude)
    {
        frequency_ = frequency;
        amplitude_ = amplitude;
    }
    void Dwmpc::startSineWave()
    {
        do_sine_wave_ = true;
    }
    void Dwmpc::stopSineWave()
    {
        do_sine_wave_ = false;
    }
    void Dwmpc::reset()
    {
        do_init_ = true;
    }
    void Dwmpc::startWalking()
    {
        timer_.startTimer();
    }
    void Dwmpc::stopWalking()
    {
        timer_.stopTimer();

    }
    void Dwmpc::setGaitParam(const double duty_factor,const double step_freq, const int gait_type)
    {   
        std::vector<double> delta;
        switch (gait_type)
        {
            case 0: //trot
                delta.push_back(0.5);
                delta.push_back(0.0);
                delta.push_back(0.0);
                delta.push_back(0.5);
                std::cout << "Trot gait selected" << std::endl;
                break;
            case 1: //crawl
                delta.push_back(0.25);
                delta.push_back(0.75);
                delta.push_back(0.98);
                delta.push_back(0.5);
                std::cout << "Crawl gait selected" << std::endl;
                break;
            case 2: //pace
                delta.push_back(0.0);
                delta.push_back(0.5);
                delta.push_back(0.0);
                delta.push_back(0.5);
                std::cout << "Pace gait selected" << std::endl;
                break;
            case 3: //jump
                delta.push_back(0.5);
                delta.push_back(0.5);
                delta.push_back(0.5);
                delta.push_back(0.5);
                std::cout << "Jump gait selected" << std::endl;
                break;
            default:
                delta.push_back(0.5);
                delta.push_back(0.0);
                delta.push_back(0.0);
                delta.push_back(0.5);
                std::cout << "Gait type not recognized, trot gait selected" << std::endl;
                break;
        }
        timer_.setDelta(delta);
        timer_.setParam(duty_factor,step_freq);
    }
    const std::map<std::string,pdata> &Dwmpc::getFullPrediction() const
    {
        return ocp_.getData();
    }
    void Dwmpc::setStepHeight(double step_height)
    {
        desired_["step_height"][0] = step_height;
    }
} //namespace controllers
