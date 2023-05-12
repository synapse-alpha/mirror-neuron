import argparse
import pickle
import pandas as pd
import torch
import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument('--offline', action='store_true', help='Run offline')
    parser.add_argument('--job_type', type=str, default='default', help='Job type')
    parser.add_argument('--group', type=str, default='default', help='Job group')
    parser.add_argument('--profile', action='store_true', help='Profile the run using pyinstrument')
    return parser.parse_args()


def save_results(path, outputs):
    outputs_cpu = []
    torch_tensor_cls = torch.tensor(1).__class__
    for out in outputs:        
        out_cpu = {
            k: v.clone().detach().to("cpu").numpy() if isinstance(v, torch_tensor_cls) and v.requires_grad else \
            v.clone().to("cpu").numpy() if isinstance(v, torch_tensor_cls) else v \
                for k,v in out.items()
        }
        outputs_cpu.append(out_cpu)
    if path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(outputs_cpu, f)
    elif path.endswith('.csv'):
        df = pd.DataFrame(outputs_cpu)
        df.to_csv(path, index=False)
    else:
        raise ValueError(f'Unknown file extension for {path!r}')

def load_results(path):
    
    print(f'Loading results from {path!r}')

    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            events = pickle.load(f)
            return pd.DataFrame(events)
    elif path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.log'):
        return parse_log(path)
    else:
        raise ValueError(f'Unknown file extension for {path!r}')
    
def parse_log(path: str) -> pd.DataFrame:
    events = []
    with open(path,'r') as f:
        for line in tqdm.tqdm(f.readlines()):
            event = json.loads(line)
            
            record = event.get('record',{})
            record_elapsed_content = record.get('elapsed',{})
            record_extra_content = record.get('extra',{})
            
            event.update(record_elapsed_content)
            event.update(record_extra_content)
            event.update({'timestamp': event['text'].split('|')[0]})
            del event['text']
            del event['record'] 
            
            events.append(event)
    
    return pd.DataFrame(events).astype({'timestamp': 'datetime64[ns]'}).reset_index().rename(columns={'index': 'step'})
