import wandb

from loaders.model import load_model
from loaders.data import load_data
from loaders.config import load_config
from query import run_query
from analysis import run_analysis

# note that you dont need to pass arguments throughout the code because wandb.config is a global variable

def main():

    # Parse the arguments
    args = parse_args()
    # check if the config file is passed
    if args.config is None:
        raise Exception("Please provide a config file using --config")

    # Load the config file
    config = load_config(args.config)

    reward_model = None
    data = None
    responses = None

    wandb.login()
    run = wandb.init(project="reward-model", entity='monkey-n', config=config)

    # Load the model
    if config.get('model'):
        reward_model = load_model(**config['model'])
        print("Model loaded")

    # Load the data
    if config.get('data'):
        data = load_data(**config['data'])
        print("Data loaded")

    # Run the queries
    if config.get('query'):
        responses = run_query(reward_model=reward_model, data=data, **config['query'])
        print("Queries executed")

    # Run the analysis
    if config.get('analysis'):
        run_analysis(reward_model=reward_model, data=data, responses=responses, **config['analysis'])
        print("Analysis executed")
    
    run.log_code()

if __name__ == "__main__":
    main()