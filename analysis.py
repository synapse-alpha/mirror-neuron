import os
import tqdm
import torch
import wandb
import numpy as np
import pandas as pd

import plotly.express as px


def run_analysis():
    pass


def run_plot(df, col, label, y='score'):

#    table = wandb.Table(data=df, columns = ["x", "y"])

#     wandb.log(
#         {f'{y}_vs_{col}' : wandb.plot.line(table, "x", "y",
#            title="Custom Y vs X Line Plot")})

    figure_dir = wandb.config.get('plots_dir','./plotly_figures')
    file_name = os.path.join(figure_dir, f'{col}_vs_{y}')
    path_to_plotly_html = f'{file_name}.html'
    # Create a table
    table = wandb.Table(columns = [file_name])

    corr = df.score.corr(df[col])
    print(f'Correlation between {col} and score: {corr:.3f}')
    fig = px.scatter(df, x=col, y='score',
        opacity=0.75,
            labels={col:label, 'y': 'Reward Score'},
            title=f'Distribution of Reward Scores: Correlation = {corr:.2f}',
            trendline='ols', marginal_y='histogram', marginal_x='histogram',
            hover_data=['question'],
            width=800, height=600, template='plotly_white')
    # fig.show()
    # fig.write_image(f'figures/{col}_vs_score.png')
    # fig.write_html(f'figures/{col}_vs_score.html')

    # Create path for Plotly figure
    fig.write_html(path_to_plotly_html, auto_play = False) 

    # Add Plotly figure as HTML file into Table
    table.add_data(wandb.Html(path_to_plotly_html))    




def run_features(df, feature_names):

    # make some high level features which describe salient properties of questions such as number of words, length of question, etc.
    if 'question_length' in feature_names:
        df['question_length'] = df.question.str.len()

    if 'num_words' in feature_names:
        df['num_words'] = df.question.str.split().apply(len)

    if 'avg_word_length' in feature_names:
        df['avg_word_length'] = df.question_length / df.num_words

    if 'median_word_length' in feature_names:
        df['median_word_length'] = df.question.str.split().apply(lambda x: np.median([len(w) for w in x]))

    if 'first_word' in feature_names:
        df['first_word'] = df.question.str.split().apply(lambda x: x[0])

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