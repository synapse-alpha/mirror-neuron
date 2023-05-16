import tqdm
import torch
import umap
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import matplotlib.pyplot as plt
from plotting.config import plotly_config

def plot_network_embedding(df, ntops=None):
    if ntops is None:
        ntops = [1, 5, 10, 50, 100, 500]
    figs = {}
    for ntop in ntops:
        d = {}
        for uid in tqdm.tqdm(df.uids.unique()):
            c = df.loc[df.uids==uid,'all_completions'].value_counts()
            d[uid] = set(c.index[:ntop])

        uids = list(d.keys())
        a = np.zeros((len(uids),len(uids)))
        for i, uid in enumerate(tqdm.tqdm(uids)):
            for j, uid2 in enumerate(uids):
                if j<i:
                    a[i,j] = a[j,j] = len( d[uid].intersection(d[uid2]) ) /ntop
                
        g = nx.from_numpy_array(a)
        z = pd.DataFrame(nx.spring_layout(g)).T.reset_index().rename(columns={0:'x',1:'y','index':'UID'})
        figs[ntop] = px.scatter(z, x='x',y='y', 
                        title=f'Graph for Top {ntop} Completion Similarities',
                        opacity=0.5,color='UID', color_continuous_scale='BlueRed',
                        **plotly_config)
    return figs

def plot_sentence_embedding(
        df,
        tokenizer=None,
        transformer=None,
        embedding_layer=None,
        reward_model=None,
        truncation=False,
        max_length=550,
        padding="max_length",
        device=None,
        scaler=None,
        x=None,
        y=None,
        z=None,
        base_column=None,
        **kwargs):

    """Adds embedding values to DF to be plotted"""
    if reward_model is not None:
        if transformer is None:
            transformer = reward_model.transformer
        if tokenizer is None:
            tokenizer = reward_model.tokenizer
        if embedding_layer is None:
            embedding_layer = reward_model.transformer.get_input_embeddings().to(device)
        if device is None:
            device = reward_model.device

    assert base_column in df.columns, f'Base column {base_column!r} not in dataframe'

    embeddings_data = []
    pbar = tqdm.tqdm(df[base_column].values, unit=base_column, desc=f'Embedding {base_column}')
    for i, question in enumerate(pbar):

        encodings_dict = tokenizer(
            ["<|startoftext|>" + question + "<|endoftext|>"],
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        )

        with torch.no_grad():
            token_embeddings = embedding_layer(encodings_dict["input_ids"].to( device ))

        embeddings_data.append({
            'question': question,
            'score': df.scores[i].item(),
            'embedding': token_embeddings[0].cpu().numpy(),
            'sentence_embedding': token_embeddings[0].mean(dim=0).cpu().numpy(),
            'input_ids': encodings_dict['input_ids'].cpu().numpy(),
            'attention_mask': encodings_dict['attention_mask'].cpu().numpy(),
        })

    df_embed = pd.DataFrame(embeddings_data)

    # Save to disk # TODO: Set this path from config
    df_embed.to_pickle('df_embed.pkl')

    # Prepare data for UMAP
    emblst = df_embed['embedding'].tolist()

    # embeddings (n_embeddings, max_len * embedding_dim)
    embeddings = np.stack(emblst).reshape([len(emblst), -1])

    # Grab list of scores
    scores = df_embed['score'].to_numpy()

    # Encode with umap-learn
    reducer = umap.UMAP(random_state=42)

    if scaler == 'MinMax':
        scaler = MinMaxScaler()
    else:
        raise "Scaler function not defined"

    scaled_scores = scaler.fit_transform(scores.reshape(-1, 1))

    colors = plt.cm.viridis(scaled_scores)
    embedding_2d = reducer.fit_transform(embeddings)

    # Compute and save the scatter plot
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=10)
    plt.colorbar(label='Model Score')
    plt.title('UMAP Scatter Plot of Sentence Embeddings vs scores (color)')
    plt.savefig('embeddings_vs_scores_umap.png')

    # TODO: (steffen/pedro) let's chat about how to incorporate this into run_plot()
    df = pd.DataFrame()
    df[x] = embedding_2d[:, 0]
    df[y] = embedding_2d[:, 1]
    df[z] = scores

    return df