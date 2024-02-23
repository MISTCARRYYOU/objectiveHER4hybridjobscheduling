
from CEDCS_Env import CEDCS_env
from agent_DRL.agents.actor_critic_agents.OTDPG import OTDPG
from agent_DRL.utilities.data_structures.Config import Config
from agent_DRL.agents.Trainer import Trainer

import torch
torch.set_num_threads(1)


config = Config()
config.seed = 1
config.use_GPU = True
embedding_dimensions = []

config.file_to_save_data_results = False
config.use_server = 0

config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = False
config.standard_deviation_results = 1.0

config.runs_per_agent = 1
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {

    "Actor_Critic_Agents": {

        "learning_rate": 0.0001,
        "linear_hidden_units": [64, 16],
        "final_layer_activation": ["sigmoid", "none"],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,
        "HER_sample_proportion": 0.8,

        "Actor": {
            "learning_rate": 0.0001,
            "linear_hidden_units": [64, 16],
            "final_layer_activation": "sigmoid",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0001,
            "linear_hidden_units": [64, 16],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 1000,
        "batch_size": 1024,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    },

    "Policy_Gradient_Agents": {
        "learning_rate": 0.0005,
        "linear_hidden_units": [64, 16],
        "final_layer_activation": "sigmoid",
        "learning_iterations_per_round": 1,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 1,  # number of CPU cores for PPO parallelly
        "normalise_rewards": True,
        "gradient_clipping_norm": 7.0,
        "mu": 0.0,  # only required for continuous action games
        "theta": 0.0,  # only required for continuous action games
        "sigma": 0.0,  # only required for continuous action games
        "epsilon_decay_rate_denominator": 1.0,
        "clip_rewards": False
    },
}

if __name__ == '__main__':

    # env = CEDCS_env('data_matrix_200.txt', 200, 200, 5, 200, 300, 200)
    config.is_train = True
    config.is_load = False

    config.load_path_epix = None  # trained model of which episodes 3000:1118-trained-model

    config.num_episodes_to_run = 100  # 2000 episodes is OK to evaluate
    config.save_fre = 200
    config.record_interval = 500  # every which epis record the transitions

    for case in [2,3,4,5]:  # 2 3 4 5
        if case == 2:
            config.environment = CEDCS_env('./instances/data_matrix_200_seed1.txt', 200, 200, 5, 200, 300, 200)
        elif case == 3:
            config.environment = CEDCS_env('./instances/data_matrix_300_seed1.txt', 300, 300, 5, 300, 300, 300)
        elif case == 4:
            config.environment = CEDCS_env('./instances/data_matrix_400_seed1.txt', 400, 400, 5, 400, 300, 400)  # seed2 3 400 seed1 300
        elif case == 5:
            config.environment = CEDCS_env('./instances/data_matrix_500_seed1.txt', 500, 500, 5, 500, 500, 500)
        config.file_to_save_results_graph = './my_data_and_graph/' + config.environment.env_name + "_epi_rews.png"
        with open('./my_data_and_graph/' + config.environment.env_name + 'logs.txt', 'w') as f:
            pass
        with open('./my_data_and_graph/losses/dnnloss.txt', 'w') as f:
            pass
        with open('./my_data_and_graph/' + config.environment.env_name + 'times.txt', 'w') as f:
            pass
        AGENTS = [OTDPG]
        # clear the logs
        for agent in AGENTS:
            with open('./my_data_and_graph/trans_' + agent.agent_name + '.txt', 'w') as f:
                pass
        trainer = Trainer(config, AGENTS)
        trainer.run_games_for_agents()
