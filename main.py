import wandb

from sources.neuron import neuron
from loaders.model import load_model
from loaders.data import load_data
from loaders.config import load_config
from query import run_query
from analysis import run_analysis
from utils import parse_args

# note that you dont need to pass arguments throughout the code because wandb.config is a global variable

def main():

    # Parse the arguments
    args = parse_args()
    # check if the config file is passed
    if args.config is None:
        raise Exception("Please provide a config file using --config")

    # Load the config file
    config = load_config(args.config)

    model = None
    data = None

    wandb.login()
    run = wandb.init(project="mirror-neuron", entity=args.entity, config=config, mode='offline' if args.offline else 'online')
    
    # capture bittensor default config and use mock wallet and subtensor
    neuron_config = neuron.config()
    neuron_config.subtensor._mock = True
    neuron_config.wallet._mock = True
    
    run.log({"neuron_config": neuron_config})

    # Load the model
    if config.get('model'):
        print(f'{"- "*40}\nLoading model:')
        model = load_model()
        run.log({"model_loaded": True})
        print("\n>>> Model loaded successfully\n")

    # Load the data
    if config.get('data'):
        print(f'{"- "*40}\nLoading query data:')
        data = load_data()
        run.log({"data_loaded": True})
        print("\n>>> Data loaded successfully\n")

    # Run the queries
    if config.get('query'):
        print(f'{"- "*40}\nRunning queries:')
        run_query(model=model, data=data)
        run.log({"queries_executed": True})
        print("\n>>> Queries executed successfully\n")

    # Run the analysis
    if config.get('analysis'):
        print(f'{"- "*40}\nRunning analysis:')
        run_analysis(model=model, data=data)
        run.log({"analysis_executed": True})
        print("\n>>> Analysis executed successfully\n")

    # run.log_code()

if __name__ == "__main__":
    main()