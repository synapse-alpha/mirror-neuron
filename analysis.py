import tqdm
import torch
import wandb
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like

from utils import load_results
from loaders.templates import AnalysisConfigTemplate

from plotting.correlation import plot_score_reward_correlation
from plotting.heatmap import run_plot_uids
from plotting.leaderboards import plot_leaderboard
from plotting.embeddings import plot_sentence_embedding
# Add plots:
# - Leaderboard of UIDs
# - Leaderboard of questions and answers
# - Stats of questions, and answers (frequencies, and rewards for doing so)
# - Completion length
# - Clustering of similar UIDs by Q&As
# - Reward stats versus block
# - Timeout rate

def sanitize(df):

    # detect list-like columns and convert to numpy arrays on cpu
    list_cols = [c for c, ser in df.items() if ser.apply(lambda x: is_list_like(x)).any()]
    for c in list_cols:
        if isinstance(df.loc[0,c], torch.Tensor):
            df[c] = df[c].apply(lambda x: x.detach().numpy())
        print(f'Column {c!r} is list-like with lengths {set(df[c].apply(len))}')
        df[c] = df[c].apply(lambda x: np.array(x).flatten())

def tabularize(df, ref_col):
    """Expand list-like columns into tabular format, using the reference column to determine the number of items to expand
    """
    # iterate over dataframe rows
    rows = []
    for idx in tqdm.tqdm(df.index):
        row = df.loc[idx]
        n = len(row[ref_col])
        rows.append( {k: v[:n] if isinstance(v, list) else v for k, v in row.items()} )
            
    list_cols = [c for c, ser in df.items() if ser.apply(lambda x: is_list_like(x)).any()]
    return pd.DataFrame(rows, index=df.index).explode(column=list_cols)


def run_analysis(model=None, data=None):

    template = AnalysisConfigTemplate(**wandb.config.analysis)
    print(f'Template: {template}')

    # load the results from the query (can be multiple)
    frames = []
    for load_path in template.requires:
        frames.append(load_results(load_path).assign(path=load_path))
    df = pd.concat(frames,axis=0)
    

    if model is not None:
        num_uids = model.metagraph.n
    else: # infer from the data
        num_uids = df.uids.apply(max).max()+1

    print(df)
    print(df.iloc[0])

    sanitize(df)

    if 'rewards' in df:
        figs = run_plot_uids(df, uid_field='uids', value_field='rewards', num_uids=num_uids)
        wandb.log({f'uid_vs_{k}': wandb.Html(fig.to_html()) for k, fig in figs.items()})

    if 'scores' in df:
        figs = run_plot_uids(df, uid_field=None, value_field='scores', num_uids=num_uids)
        wandb.log({f'uid_vs_{k}': wandb.Html(fig.to_html()) for k, fig in figs.items()})
        
        wandb.log({'correlation': plot_score_reward_correlation(df)})    
    
    # overwrite dataframe with long format (plays nicer with plotly)
    df = tabularize(df)     
       
    # save dataframe to wandb as a table 
    # wandb.log({'history': wandb.Table(dataframe=df)})

    print(df)
    print(df.iloc[0])
    
    if 'step' not in df:
        df['step'] = np.arange(len(df))
    if 'step_sub' not in df:
        df['step_sub'] = np.arange(len(df))
        

    if 'call_time' in df:
        run_plot(df, x='step', y='call_time', title='Call time', type='line')
    
    if template.create_features:
        for base_column, feature_names in template.create_features.items():
            run_features(df=df, feature_names=feature_names, model=model, base_column=base_column)

    if template.plot is not None:
        for y, x_list in template.plot.items():
            for x in x_list:
                run_plot(df, x=x, y=y)


    if 'embeddings' in template.create_features:
        embedding_scaler = template.embedding_plot['scaler']
        columns = template.embedding_plot['columns']
        x = columns[0]
        y = columns [1]
        z = columns[2]

        embedded_df = plot_sentence_embedding(df, reward_model=model.reward_model, scaler=embedding_scaler, x=x, y=y, z=z, prompt_name=template.prompt_name)

        run_plot3d(embedded_df, x, y, z)


def run_plot(df, x, y, type='scatter', title=None, **kwargs):
    table = wandb.Table(data=df[[x,y]], columns = [x,y])

    plot_func = getattr(wandb.plot, type, None)
    if plot_func is None:
        raise ValueError(f'Unknown plot type {type!r}')

    if title is None:
        title = f'{x} vs {y}'

    wandb.log(
        {title : plot_func(table, x, y,
            title=title, **kwargs)})

def run_line_series(xs, ys, keys, title):
    wandb.log(
        {title : wandb.plot.line_series(xs=xs, ys=ys, keys=keys,
            title=title)})


# TODO: Evaluate if plot3d config is ok in order to modify run_plot function to support Z (removing run_plot3d)
def run_plot3d(df, x, y, z):
    table = wandb.Table(data=df[[x,y,z]], columns=[x,y,z])
    wandb.log(
        {f'{x}_and_{y}_vs_{z}' : wandb.plot.scatter(table, x, y,
            title=f'{x} vs {y}')})


def run_features(df, feature_names, base_column, model=None):

    assert base_column in df.columns, f'Base column {base_column!r} not in dataframe'
    
    # make some high level features which describe salient properties of questions such as number of words, length of question, etc.
    if 'sentence_length' in feature_names:
        df['sentence_length'] = df[base_column].str.len()

    if 'num_words' in feature_names:
        df['num_words'] = df[base_column].str.split().apply(len)

    if 'avg_word_length' in feature_names:
        df['avg_word_length'] = df.question_length / df.num_words

    if 'median_word_length' in feature_names:
        df['median_word_length'] = df[base_column].str.split().apply(lambda x: np.median([len(w) for w in x]))

    if 'first_word' in feature_names:
        df['first_word'] = df[base_column].str.split().apply(lambda x: x[0])

    if 'reward_num_tokens' in feature_names:
        assert model is not None, f'A model with reward_model attribute is required to compute the number of tokens'
        df['reward_num_tokens'] = df[base_column].apply(lambda x: len(model.reward_model.tokenizer.encode(x)))

    return df

