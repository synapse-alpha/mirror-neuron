import pandas as pd
import plotly.express as px

def plot_score_reward_correlation(df):
    df['successful_scores' ] = df.apply(lambda x: x['scores'][x['uids']],axis=1)
    long_df = df[['rewards','successful_scores']].explode(['rewards','successful_scores']).astype(float)    
    r_p = long_df.groupby(long_df.index).corr(method='pearson')
    cc_p = r_p.loc[r_p.index.get_level_values(1) == 'rewards',['successful_scores']].reset_index(drop=True).rolling(10).mean()    
    
    r_s = long_df.groupby(long_df.index).corr(method='spearman')
    cc_s = r_s.loc[r_s.index.get_level_values(1) == 'rewards',['successful_scores']].reset_index(drop=True).rolling(10).mean()
    df_corr = pd.concat([cc_p.assign(method='pearson'), cc_s.assign(method='spearman')])
    return px.line(df_corr, y='successful_scores', color='method', 
                  title='Correlation of Scores and Rewards, Rolling 10 Iterations',
                  labels={'successful_scores': 'Correlation', 'index': 'Iteration'},
                  width=800, height=600, template='plotly_white'
                  ).update_traces(opacity=0.7
                ).update_layout(legend_title_text='Method',font_size=14)
