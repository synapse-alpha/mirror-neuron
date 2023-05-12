
from plotting.config import plotly_config
import plotly.express as px

def plot_leaderboard(df, group_on='uids', col='rewards', agg='mean', ntop=20):
    """Expects long-style dataframe with no list-like columns
    """
    
    rankings = df.groupby(group_on)[col].agg(agg).sort_values().tail(ntop)
    return px.bar(x=rankings, y=rankings.index.astype(str), 
        color=rankings, orientation='h',
        labels={'x':f'{col.title()}','y':group_on, 'color':''}, 
        title=f'{col.title()} Leaderboard, top {ntop} {group_on}',
        color_continuous_scale='BlueRed',
        opacity=0.5,
        **plotly_config)
    