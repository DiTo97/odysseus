# ESBDQN: E-scooter Smart Balancing DQN

This repo is the JAX implementation based on DeepMind's [DQN library](https://github.com/deepmind/dqn_zoo) of our WOA 2021 paper on ["Smart Balancing of E-scooter Sharing Systems via Deep Reinforcement Learning"](http://ceur-ws.org/Vol-2963/paper16.pdf), developed by Federico Minutoli, Gianvito Losapio, Viviana Mascardi, and Angelo Ferrando at the University of Genoa.

**Please note:** This repo uses a custom version of the [ODySSEUS simulator](https://odysseus-simulator.readthedocs.io/en/latest/index.html) developed by the SmartData lab at PoliTO. More information on the changes we have made can be found in the paper.

## Usage guide

Follow these instructions to try out our code.

### Data

Data for the municipality of Louisville is available for download [here](https://mega.nz/file/fNQFiYBQ#RoWBIreahyKmthdB0mo4z8Sfsou-MjGQhbyEv4OL_lQ) on MEGA.
- Unzip the folder and put its contents within `odysseus\city_data_manager\data`.

Demand model of Louisville is available for download likewise [here](https://mega.nz/file/eRR3AShS#WE3tnhpSpvDJskHn-jXHNxMXcXqGNLmBrB-mb6TVgL4) on MEGA.
- Unzip the folder and put its contents within `odysseus\demand_modelling\demand_models`.

Data and demand model for the municipality of Austin are available upon request.

### Input parameters

Input parameters take into account both simulation parameters of the ODySSEUS environment, as well as, paramters for the training/test of the Rainbow agents.

#### Simulation parameters

Simulation parameters can be found in the folder `esbdqn\configs\escooter_mobility` under the name `sim_conf_<City>.py` with identical structure. Each file comprises two Python objects, named `General` and `Multiple_runs`.

**General object**

- _city_, name of the city, either Louisville or Austin;
- _relocation_workers_working_hours_, shift hours for relocation workers;
- _bin_side_length_, side length of the square zones each operative area is split into;
- _year_, year of the trip requests to consider;
- _month_start_, _month_end_, start and end of the month of the trip requests to consider;
- _day_start_, _day_end_, start and end of the day of the trip requests to consider;
- _save_history_, whether to save the results CSV after each iteration.

**Multiple_runs object**

- _n_vehicles_, number of vehicles to spawn in the environment;
- _incentive_willingness_, acceptance probability for each incentive proposal;
- _alpha_, threshold on the battery level to mark vehicles as out-of-charge in percentage between 0 and 100;
- _battery_swap_, toggle for battery swap events in the environment, either True or False;
- _n_workers_, number of battery swap workers;
- _battery_swap_capacity_, maximum number of vehicles each battery swap worker can process hourly;
- _scooter_relocation_, toggle for relocation events in the environment, either True or False;
- _n_relocation_workers_, number of relocation workers;
- _relocation_capacity_, maximum number of vehicles each relocation worker can move hourly;

All the parameters that have not been touched or are unused with respect to the original ODySSEUS simulator have been omitted above.

#### Agents parameters

Agents parameters can be found in the file `esbdqn\train.py`. Also, they can be submitted at runtime when launching `esbdqn\train.py` via CLI.

- _learning_rate_, learning rate of the Adam optimizer;
- _learn_period_, learning period of the Rainbow agents;
- _batch_size_, batch size of the networks withing the agents;
- _n_steps_, how many steps to look in the past when agents take decisions;
- _max_global_grad_norm_, global gradient norm clipping of the networks weights;
- _importance_sampling_exponent_begin_value_, _importance_sampling_exponent_end_value_, range of the importance sampling exponent;
- _replay_capacity_, experience replay buffer capacity. Should amount to about 30 repetitions of any given day;
- _priority_exponent_, priority of the timesteps stored in the experience replay buffer;
- _target_network_update_period_, update period from the online network to the offline network within each agent;
- _num_iterations_, number of training iterations;
- _max_steps_per_episode_, number of trips per episode;
- _num_eval_frames_, total number of validation trips per iteration;
- _num_train_frames_, total number of training trips per iteration (Should be at least double the validation trips);
- _n_lives_, total number of lives per iteration. Defaults to 50.

#### Experiment parameters

Each call of `esbdqn\train.py` can be named as a different experiment with its own checkpoints.

- _exp_name_, name of the experiment directory.
- _checkpoint_, toggle on whether to store a checkpoint, either True or False.
- _checkpoint_period_, period of storage of a new training checkpoint.

### Output

Run `esbdqn\train.py` to train a new ESBDQN model from scratch. Otherwise, to train starting from a checkpoint, set the _checkpoint_ toggle to True, and ensure that there is a checkpoint within the experiment directory in the form: `<Experiment_dir>\models\ODySSEUS-<City>`.

Results of each run will be stored as CSV files within the automatically generated directory `<Experiment_dir>\results`.

To reproduce the experiments in the paper:
- Set _incentive_willingness_ to 0 to obtain all the _No incentives_ data.
- Set _incentive_willingness_ to 1 and track the columns _eval_avg_pct_satisfied_demand_ and _train_avg_pct_satisfied_demand_ from the CSV files for the _Validation_ and _Training_ data, respectively.

All our experiments have been run on Ubuntu 18.04.
