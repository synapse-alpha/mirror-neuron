import pandas as pd
import numpy as np
import plotly.express as px

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