import os
from datetime import datetime
import itertools
from pathlib import Path

def get_save_path():
    return "result/hyperparameter_search/"
    

# make sure to put the default value first, for edge search
hp_dict = {
    "--k": [1],
    "--batch_size": [8, 64, 256],
    "--load_from_file": ["result/embedding_english_prompts.pkl", "result/embedding_self_prompts.pkl"],
    "--subtract_means": [False, True],
    "--contrastive_learning": [True],
    "--test_split": [0.3],
    "--mlp_n_hidden": [0, 1, 2],
    "--mlp_hidden_dim": [1536, 100], 
    "--mlp_output_dim": [3072, 1000, 100],
    "--mlp_train_epochs": [20],
    "--contrastive_loss_positive_coef": [2, 8, 16],
    "--contrastive_loss_margin": [0.5],
    "--contrastive_loss_C": [0.5, 0.2, 0.8],
    "--save_path": [get_save_path()],
}

test_dict = {
    'a': [1,2],
    'b': [1,2],
    'c': [1],
    'd': [2],
}


base_command = "python main.py "

def calc_n_runs(hp_dict):
    n_runs = 1
    for key in hp_dict.keys():
        n_runs *= len(hp_dict[key])
    return n_runs

def keyword_get_options(keyword, hp_dict):
    options = [] 
    for value in hp_dict[keyword]:
        if isinstance(value, bool):
            options.append(keyword + " " if value else "")
        else:
            if (keyword == "--load_from_file") and ("self_prompts" in value):
                options.append(f'{keyword} {value} --self_prompts ')
            else: 
                options.append(f'{keyword} {value} ')
     
    return options

def keyword_get_default(keyword, hp_dict):
    return [keyword_get_options(keyword, hp_dict)[0]]

def keyword_get_num_options(keyword, hp_dict):
    return len(hp_dict[keyword])
    
def commands_grid_search(hp_dict):
    print(f'About to execute {calc_n_runs(hp_dict)} runs...')
    command_blocks = [[base_command]]
    for keyword in hp_dict.keys():
        command_blocks.append(keyword_get_options(keyword, hp_dict))
    command_combos = list(itertools.product(*command_blocks))
    commands = []
    for keyword_list in command_combos:
        commands.append("".join(keyword_list))
    return commands

def commands_default_run(hp_dict):
    '''
    For each of the values, this chooses the default value
    '''
    command = base_command
    for keyword in hp_dict.keys():
        command += keyword_get_default(keyword, hp_dict)[0]
    return [command]

def commands_almost_default_run(keyword, i, hp_dict):
    command = base_command
    command += keyword_get_options(keyword, hp_dict)[i]
    for keyword_other in hp_dict:
        if keyword_other == keyword:
            continue
        command += keyword_get_default(keyword_other, hp_dict)[0]
    return [command]

def commands_vary_one_keyword(keyword, hp_dict):
    commands = []
    for i in range(keyword_get_num_options(keyword, hp_dict)):
        commands.append(commands_almost_default_run(keyword, i, hp_dict)[0])
    return commands
    
def commands_edge_search(hp_dict): 
    '''
    For each keyword, it tries each value, 
    while keeping the rest of the values in the default position (first option)
    '''
    commands = commands_default_run(hp_dict)
    for varying_keyword in hp_dict.keys():
        if keyword_get_num_options(varying_keyword, hp_dict) > 1:
            for i in range(1, keyword_get_num_options(varying_keyword, hp_dict)):
                commands.append(commands_almost_default_run(varying_keyword, i, hp_dict)[0])
    return commands


    
def run_commands(commands):
    print(f'running {len(commands)} experiments...')
    save_path = commands[0].split("--save_path")[1].strip().split(" ")[0] 
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    for i, command in enumerate(commands): 
        print(f'__________________________{i+1}/{len(commands)}_____________________________')
        with open(save_path + 'metrics.txt', 'a') as file:
            file.write('___________________________________________________\n')
            file.write(str(command) + '\n')
        os.system(command)

        

# change this to run something else
run_commands(
    commands_edge_search(hp_dict)
)

# run_commands(
#     commands_grid_search(hp_dict)
# )