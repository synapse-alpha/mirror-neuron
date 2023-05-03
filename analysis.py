import os
import tqdm
import torch
import wandb
import numpy as np
import pandas as pd
from utils import load_results

from loaders.templates import AnalysisConfigTemplate, QueryConfigTemplate

from pandas.api.types import is_list_like
import plotly.express as px

def _tabularize_df_hack(df):
    """Convert list-like columns to tabular format
    """
    # detect list-like columns
    list_cols = [c for c, ser in df.items() if ser.apply(lambda x: is_list_like(x)).any()]
    for c in list_cols:
        if isinstance(df.loc[0,c], torch.Tensor):
            df[c] = df[c].apply(lambda x: x.detach().numpy())
            
        # here we just grab first index of the list (TEMPORARY HACK)
        df[c] = df[c].apply(lambda x: np.array(x).flatten()[0])


def run_analysis(model=None, data=None):

    template = AnalysisConfigTemplate(**wandb.config.analysis)
    print(f'Template: {template}')

    load_path = QueryConfigTemplate(**wandb.config.query).save_path()

    df = load_results(load_path)

    # we want to unwrap the dataframe but we need to know which columns have lists and also how long the lists are
    # df = unwrap_df(df)
    print(df)
    print(df.iloc[0])
        
    _tabularize_df_hack(df) # just take first element of everything list like
    print(df)
    print(df.iloc[0])

    run_features(df, template.create_features)

    if template.plot is not None:
        for y, x_list in template.plot.items():
            for x in x_list:
                run_plot(df, x=x, y=y)


def run_plot(df, x, y='score'):

    table = wandb.Table(data=df[[x,y]], columns = [x,y])

    wandb.log(
        {f'{x}_vs_{y}' : wandb.plot.scatter(table, x, y,
            title=f'{x} vs {y}')})




def run_features(df, feature_names, col='message'):

    # make some high level features which describe salient properties of questions such as number of words, length of question, etc.
    if 'question_length' in feature_names:
        df['question_length'] = df[col].str.len()

    if 'num_words' in feature_names:
        df['num_words'] = df[col].str.split().apply(len)

    if 'avg_word_length' in feature_names:
        df['avg_word_length'] = df.question_length / df.num_words

    if 'median_word_length' in feature_names:
        df['median_word_length'] = df[col].str.split().apply(lambda x: np.median([len(w) for w in x]))

    if 'first_word' in feature_names:
        df['first_word'] = df[col].str.split().apply(lambda x: x[0])

    return df

def run_sentence_embedding(df, tokenizer=None, transformer=None, embedding_layer=None, reward_model=None, truncation=False, max_length=550, padding="max_length", device=None,**kwargs):

    if reward_model is not None:
        if transformer is None:
            transformer = reward_model.transformer
        if tokenizer is None:
            tokenizer = reward_model.tokenizer
        if embedding_layer is None:
            embedding_layer = reward_model.transformer.get_input_embeddings()
        if device is None:
            device = reward_model.device

    embeddings = []
    scores = df.response.values
    pbar = tqdm.tqdm(df.questions.values, unit='questions', desc='Embedding questions')
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

        embeddings.append({
            'question': question,
            'score': scores[i].item(),
            'embedding': token_embeddings[0].cpu().numpy(),
            'sentence_embedding': token_embeddings[0].mean(dim=0).cpu().numpy(),
            'input_ids': encodings_dict['input_ids'].cpu().numpy(),
            'attention_mask': encodings_dict['attention_mask'].cpu().numpy(),
            })

    df_embed = pd.DataFrame(embeddings)
    df_embed.head()


def run_preprocessing(X, y, type, **kwargs):

    if type.lower() == 'pca':
        from sklearn.decomposition import PCA

        # try dimensional reduction
        pca = PCA().fit(X)

        evr_cumsum = pca.explained_variance_ratio_.cumsum()
        fig = px.line(y=evr_cumsum,
                labels={'x':'Principal Component',
                        'y':'Explained Variance Ratio'},
                template='plotly_white',
                title='Sentence Embeddings PCA Dimensional Reduction',
                )

        n_components = kwargs.get('n_components')
        if n_components:
            if 0 < n_components < 1:
                n_components = int(n_components*X.shape[1])

        threshold = kwargs.get('threshold')
        if threshold:
            n_components = np.where(evr_cumsum > threshold)[0][0]

        fig.add_vline(x=n_components, line_dash='dash', line_color='red')

        # transform sentence embeddings into n_components-dim PCA components.
        return pca.transform(X)[:,:n_components]
    else:
        raise NotImplementedError(f'Preprocessing type {type!r} not implemented.')


def run_estimator(X, y, type, **kwargs):

    from sklearn.model_selection import train_test_split
    if type == 'GradientBoostingRegressor':
        from sklearn.ensemble import GradientBoostingRegressor

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        gbr = GradientBoostingRegressor(**kwargs).fit(X_train, y_train)

        preds = gbr.predict(X)
        df_model = pd.DataFrame({'y':y, 'y_pred':preds, 'abs_error':np.abs(preds-y)})
        df_model['subset'] = 'train'
        df_model.loc[y_test.index, 'subset'] = 'test'

        print(f'Mean absolute error: {df_model.abs_error.mean():.3f}')
        print(f'Model training score: {gbr.score(X_train, y_train):.3f}, test score: {gbr.score(X_test, y_test):.3f}')

        px.line(y=gbr.train_score_,
                labels={'x':'Number of Iterations',
                        'y':'MSE Loss'},
                template='plotly_white',
                title='GradientBoostingRegressor Training Loss',
                ).show()

        if hasattr(gbr, 'oob_improvement_'):
            px.line(y=gbr.oob_improvement_,
                    labels={'x':'Number of Iterations',
                            'y':'OOB Improvement'},
                    template='plotly_white',
                    title='GradientBoostingRegressor Validation Improvement',
                    ).show()

        ymin, ymax = df_model.y.min(), df_model.y.max()

        fig = px.scatter(df_model, x='y', y='y_pred', color='abs_error', facet_col='subset',
                template='plotly_white', color_continuous_scale='BlueRed',
                opacity=0.3)
        # add diagonal line at y=x usig fig.add_shape() and apply to all subplots
        fig.add_shape(x0=ymin, y0=ymin, x1=ymax, y1=ymax, line=dict(color='black', width=1), type='line', row=1, col=1)
        fig.add_shape(x0=ymin, y0=ymin, x1=ymax, y1=ymax, line=dict(color='black', width=1), type='line', row=1, col=2)