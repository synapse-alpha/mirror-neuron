import tqdm
import torch
import umap

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def run_sentence_embedding(
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