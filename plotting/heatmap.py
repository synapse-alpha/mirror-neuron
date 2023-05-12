import plotly.express as px
import pandas as pd
import numpy as np
from plotting.config import plotly_config

def plot_heatmap(df, x, y, z, title=None, **kwargs):
    return px.density_heatmap(df, x=x, y=y, z=z, title=title, **kwargs)


def run_plot_uids(df, uid_field, value_field, num_uids):
    """Expects wide-style dataframe with list-like columns"""

    arrs = []
    for i in range(df.shape[0]):
        # make an empty array full of nans
        arr = np.full(num_uids, np.nan)
        # TODO: in train mode we have 2 forward passes per step, so we want to make a separate panel for each :)
        row = df.iloc[i]

        uids = row[uid_field] if uid_field is not None else range(num_uids)

        arr[uids] = row[value_field]
        arrs.append(arr)

    df_uids = pd.DataFrame(arrs).T
    figs = {}
    figs[f'uid_vs_{value_field}'] = px.imshow(df_uids,
              title=f'{value_field.title()} for Network',
              color_continuous_scale='Blues', aspect='auto',
              **plotly_config).\
        update_xaxes(title='Iteration').\
        update_yaxes(title='UID').\
        update_layout(font_size=14)

    figs[f'uid_vs_{value_field}_ewm'] = px.imshow(df_uids.ewm(alpha=0.1, axis=1).mean(),
              title=f'{value_field.title()} for Network (EWM Smoothed)',
              color_continuous_scale='Blues', aspect='auto',
              width=800, height=600, template='plotly_white').\
        update_xaxes(title='Iteration').\
        update_yaxes(title='UID').\
        update_layout(font_size=14)

    avg_values = df_uids.mean(axis=1)
    figs[f'average_uid_{value_field}'] = px.bar(x=avg_values.index, y=avg_values.values, color=avg_values.values,
                 error_y=df_uids.std(axis=1).values,
                 color_continuous_scale='Blues',
                title=f'Average {value_field.title()} by UID',
                 labels={'x': 'UID', 'y': f'Average {value_field.title()}','color':''},
                 width=800, height=600, template='plotly_white',
        ).update_traces(marker_line_color='grey'
        ).update_layout(font_size=14,coloraxis_showscale=False)
    # change error bar color to grey
    figs[f'average_uid_{value_field}'].data[0].error_y.color = 'grey'

    figs[f'average_{value_field}'] = px.histogram(x=avg_values,
                title=f'Average {value_field.title()}, all UIDs',
                 labels={'x': f'Average {value_field.title()}','color':''},
                 width=800, height=600, template='plotly_white',
        ).update_layout(font_size=14,coloraxis_showscale=False)

    # add quantiles and draw a stacked histogram for each
    steps = df_uids.columns.astype(float)
    n_bins = 5
    labels = [f'{i} - {i+100//n_bins}%' for i in range(0, 100, 100//n_bins)]
    cats = pd.cut(steps,bins=n_bins,labels=labels)
    figs[f'average_slices_{value_field}'] = px.histogram(df_uids.groupby(cats, axis=1).mean(), opacity=0.8,
                labels={'value': f'Average {value_field.title()}','variable':'Step Quantile'},
                title=f'Average {value_field.title()}, all UIDs, by Quantile',
                 width=800, height=600, template='plotly_white',
                 color_discrete_sequence=px.colors.sequential.deep_r
            ).update_layout(font_size=14,coloraxis_showscale=False)

    hit_success = df.uids.explode().value_counts().sort_index()
    figs[f'uid_hit_success'] = px.bar(x=hit_success.index, y=hit_success.values, color=hit_success,
           labels={'x':'UID','y':'Number of Calls'},
           title='Number of Successful Calls by UID',
           color_continuous_scale='Blues', height=600, width=800, template='plotly_white'
        ).update_traces(marker_line_color='grey'
        ).update_layout(font_size=14, coloraxis_showscale=False)
    
    hit_total = df.all_uids.explode().value_counts().sort_index()
    figs[f'uid_hit_total'] = px.bar(x=hit_total.index, y=hit_total.values, color=hit_total,
              labels={'x':'UID','y':'Number of Calls'},
              title='Total Number of Calls by UID',
              color_continuous_scale='Blues', height=600, width=800, template='plotly_white'
        ).update_traces(marker_line_color='grey'
        ).update_layout(font_size=14, coloraxis_showscale=False)

    return figs    
