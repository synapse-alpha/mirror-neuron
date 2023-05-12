
from plotting.config import plotly_config
import plotly.express as px
import pandas as pd

def plot_throughput(df, time_col='timestamp', rule='1H'):
    event_rate = df.resample(on=time_col, rule=rule).size()
    event_rate = event_rate.reset_index().rename(columns={0:'Count'})
    return px.line(event_rate, x=time_col, y='Count',
            title='Throughput',
            labels={'x':'','y':f'Events / {rule}'}, title='Event Rate',
            **plotly_config).update_traces(opacity=0.7)
    
    
def plot_dendrite_rates(df, ntop=20):
    
    top_uids = df.all_uids.value_counts().head(ntop).index
    hit_total = df.all_uids.explode().value_counts().sort_index().loc[top_uids]
    hit_success = df.uids.explode().value_counts().sort_index().loc[top_uids]    

    all_rates = pd.concat([hit_total, hit_success], axis=1).rename(columns={'all_uids':'Total','uids':'Success',0:'Rate'})
    long_rates = all_rates.melt(var_name='Type', ignore_index=False).reset_index().rename(columns={'index':'UID'})
    fig = px.bar(long_rates.astype({'UID':str}),
                 x='value', y='UID', color='Type',
            labels={'x':'Number of Calls'},
            barmode='group',
            title='Dendrite Calls by UID',
            color_continuous_scale='Blues', **plotly_config
        ).update_traces(marker_line_color='grey'
        ).update_layout(font_size=14, coloraxis_showscale=False)
    fig.show()


def plot_message_rates(df, msg_col='all_completions', time_interval='day', ntop=20):
    if hasattr(df.timestamp.dt, time_interval):
        timestamp = getattr(df.timestamp.dt, time_interval)
        
    top_completions = df.completion.value_counts().head(ntop)
    c = df.groupby([msg_col,timestamp]).size()
    cc = c.loc[top_completions.index].reset_index().rename(columns={0:'Count'})
    cc['Message ID'] = cc[msg_col].map({k:f'Msg {i}: {top_completions.loc[k]}' for i, k in enumerate(top_completions.index)})
    return px.line(cc, x='timestamp', y='Count',color='Message ID', 
            hover_data=[df[msg_col].str.replace('\n','<br>')],
            title='Message Rate', line_shape='spline',
            labels={'timestamp':f'Time, {time_interval}', 'Count':'Occurrences'}, 
            **plotly_config).update_traces(opacity=0.7)    
    
