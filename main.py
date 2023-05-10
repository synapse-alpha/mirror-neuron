import os
import wandb
import bittensor
import pyinstrument
from sources.neuron import neuron
from loaders.model import load_neuron
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

    tags = [k for k in ["model", "data", "query", "analysis"] if config.get(k)]

    profiler = None
    model = None
    data = None

    if args.profile:
        profiler = pyinstrument.Profiler()
        print(f'{"*"*40}\nProfiling run using pyinstrument\n{"*"*40}')
        profiler.start()
        tags += ["profiler"]

    wandb.login()
    run = wandb.init(
        project="mirror-neuron",
        entity="mirror-neuron",
        config=config,
        mode="offline" if args.offline else "online",
        job_type=args.job_type,
        group=args.group,
        tags=tags,
    )

    # capture bittensor default config and use mock wallet and subtensor
    bt_config = neuron.config()
    # bt_config.subtensor._mock = True
    bt_config.wallet._mock = True
    bt_config.neuron.dont_save_events = True
    bt_config.neuron.device = config.get(
        "device", "cpu"
    )  # ensure these are synchronized from main config.yml
    # remove unwanted extra config fields
    for k in ["model", "data", "query", "analysis"]:
        if hasattr(bt_config, k):
            delattr(bt_config, k)

    run.config.update({"bittensor": bt_config})

    # Load the model
    if config.get("model"):
        print(f'{"- "*40}\nLoading model:')
        model = load_neuron(bt_config=bt_config)
        print("\n>>> Model loaded successfully\n")

    # Load the data
    if config.get("data"):
        print(f'{"- "*40}\nLoading query data:')
        data = load_data()
        print("\n>>> Data loaded successfully\n")

    # Run the queries
    if config.get("query"):
        print(f'{"- "*40}\nRunning queries:')
        run_query(model=model, data=data)
        print("\n>>> Queries executed successfully\n")

    # Run the analysis
    if config.get("analysis"):
        print(f'{"- "*40}\nRunning analysis:')
        run_analysis(model=model, data=data)
        print("\n>>> Analysis executed successfully\n")

    if profiler:
        profiler.stop()
        profile_report = profiler.output_html(timeline=True)
        path = f"./profiles/{run.name}.html"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(profile_report)
        wandb.log({"pyinstrument": wandb.Html(profile_report)})


if __name__ == "__main__":
    main()
