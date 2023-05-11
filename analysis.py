import os
import tqdm
import umap
import torch
import wandb
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from pandas.api.types import is_list_like
from sklearn.preprocessing import MinMaxScaler

from utils import load_results
from loaders.templates import AnalysisConfigTemplate, QueryConfigTemplate


def sanitize(df):

    # detect list-like columns and convert to numpy arrays on cpu
    list_cols = [
        c for c, ser in df.items() if ser.apply(lambda x: is_list_like(x)).any()
    ]
    for c in list_cols:
        if isinstance(df.loc[0, c], torch.Tensor):
            df[c] = df[c].apply(lambda x: x.detach().numpy())
        print(f"Column {c!r} is list-like with lengths {set(df[c].apply(len))}")
        df[c] = df[c].apply(lambda x: np.array(x).flatten())


def _tabularize_hack(df):
    """Convert list-like columns to tabular format
    """
    # detect list-like columns
    list_cols = [
        c for c, ser in df.items() if ser.apply(lambda x: is_list_like(x)).any()
    ]
    for c in list_cols:
        # here we just grab first index of the list (TEMPORARY HACK)
        df[c] = df[c].apply(lambda x: x[0])


def run_analysis(model=None, data=None):

    template = AnalysisConfigTemplate(**wandb.config.analysis)
    print(f"Template: {template}")

    # load the results from the query
    load_path = QueryConfigTemplate(**wandb.config.query).save_path()
    df = load_results(load_path)

    if model is not None:
        num_uids = model.metagraph.n
    else:  # infer from the data
        num_uids = df.uids.apply(max).max()

    print(df)
    print(df.iloc[0])

    sanitize(df)

    df["successful_scores"] = df.apply(lambda x: x["scores"][x["uids"]], axis=1)
    long_df = (
        df[["rewards", "successful_scores"]]
        .explode(["rewards", "successful_scores"])
        .astype(float)
    )
    r_p = long_df.groupby(long_df.index).corr(method="pearson")
    cc_p = (
        r_p.loc[r_p.index.get_level_values(1) == "rewards", ["successful_scores"]]
        .reset_index(drop=True)
        .rolling(10)
        .mean()
    )

    r_s = long_df.groupby(long_df.index).corr(method="spearman")
    cc_s = (
        r_s.loc[r_s.index.get_level_values(1) == "rewards", ["successful_scores"]]
        .reset_index(drop=True)
        .rolling(10)
        .mean()
    )
    df_corr = pd.concat([cc_p.assign(method="pearson"), cc_s.assign(method="spearman")])
    fig = (
        px.line(
            df_corr,
            y="successful_scores",
            color="method",
            title="Correlation of Scores and Rewards, Rolling 10 Iterations",
            labels={"successful_scores": "Correlation", "index": "Iteration"},
            width=800,
            height=600,
            template="plotly_white",
        )
        .update_traces(opacity=0.7)
        .update_layout(legend_title_text="Method", font_size=14)
    )
    wandb.log({"correlation": fig})

    run_plot_uids(df, uid_field="uids", value_field="rewards", num_uids=num_uids)
    run_plot_uids(df, uid_field=None, value_field="scores", num_uids=num_uids)
    # save dataframe to wandb as a table (this doesn't work because of the list-like columns)
    # wandb.log({'history': wandb.Table(dataframe=df)})

    _tabularize_hack(df)  # just take first element of everything list like

    # save dataframe to wandb as a table
    wandb.log({"history_tabular": wandb.Table(dataframe=df)})

    print(df)
    print(df.iloc[0])

    rewards_table = df.set_index(["step", "step_sub"])["rewards"].apply(pd.Series)
    rewards_table.columns = [f"reward_{i}" for i in range(rewards_table.shape[1])]
    run_line_series(
        xs=[df.step.values for _ in rewards_table.columns],
        ys=[rewards_table[c].values for c in rewards_table.columns],
        keys=rewards_table.columns,
        title="Rewards",
    )

    run_plot(df, x="step", y="call_time", title="Call time", type="line")
    run_features(df=df, feature_names=template.create_features, model=model)

    if template.plot is not None:
        for y, x_list in template.plot.items():
            for x in x_list:
                run_plot(df, x=x, y=y)

    if "embeddings" in template.create_features:
        embedding_scaler = template.embedding_plot["scaler"]
        columns = template.embedding_plot["columns"]
        x = columns[0]
        y = columns[1]
        z = columns[2]

        embedded_df = run_sentence_embedding(
            df, reward_model=model.reward_model, scaler=embedding_scaler, x=x, y=y, z=z
        )

        run_plot3d(embedded_df, x, y, z)


def run_plot_uids(df, uid_field, value_field, num_uids):

    arrs = []
    for i in range(df.shape[0]):
        # make an empty array full of nans
        arr = np.full(num_uids, np.nan)
        # TODO: in train mode we have 2 forward passes per step, so we want to make a separate panel for each :)
        row = df.iloc[i]

        uids = row[uid_field] if uid_field is not None else range(num_uids)

        arr[uids] = row[value_field]
        arrs.append(arr)
        # wandb.log({value_field: wandb.Histogram(arr)}, step=row.step)

    df_uids = pd.DataFrame(arrs).T
    fig = (
        px.imshow(
            df_uids,
            title=f"{value_field.title()} for Network",
            color_continuous_scale="Blues",
            aspect="auto",
            width=800,
            height=600,
            template="plotly_white",
        )
        .update_xaxes(title="Iteration")
        .update_yaxes(title="UID")
        .update_layout(font_size=14)
    )
    wandb.log({f"uid_vs_{value_field}": wandb.Html(fig.to_html())})

    fig = (
        px.imshow(
            df_uids.ewm(alpha=0.1, axis=1).mean(),
            title=f"{value_field.title()} for Network (EWM Smoothed)",
            color_continuous_scale="Blues",
            aspect="auto",
            width=800,
            height=600,
            template="plotly_white",
        )
        .update_xaxes(title="Iteration")
        .update_yaxes(title="UID")
        .update_layout(font_size=14)
    )
    wandb.log({f"uid_vs_{value_field}_ewm": wandb.Html(fig.to_html())})

    avg_values = df_uids.mean(axis=1)
    fig = (
        px.bar(
            x=avg_values.index,
            y=avg_values.values,
            color=avg_values.values,
            error_y=df_uids.std(axis=1).values,
            color_continuous_scale="Blues",
            title=f"Average {value_field.title()} by UID",
            labels={"x": "UID", "y": f"Average {value_field.title()}", "color": ""},
            width=800,
            height=600,
            template="plotly_white",
        )
        .update_traces(marker_line_color="grey")
        .update_layout(font_size=14, coloraxis_showscale=False)
    )
    # change error bar color to grey
    fig.data[0].error_y.color = "grey"

    wandb.log({f"average_uid_{value_field}": wandb.Html(fig.to_html())})

    fig = px.histogram(
        x=avg_values,
        title=f"Average {value_field.title()}, all UIDs",
        labels={"x": f"Average {value_field.title()}", "color": ""},
        width=800,
        height=600,
        template="plotly_white",
    ).update_layout(font_size=14, coloraxis_showscale=False)

    wandb.log({f"average_{value_field}": wandb.Html(fig.to_html())})

    # add quantiles and draw a stacked histogram for each
    steps = df_uids.columns.astype(float)
    n_bins = 5
    labels = [f"{i} - {i+100//n_bins}%" for i in range(0, 100, 100 // n_bins)]
    cats = pd.cut(steps, bins=n_bins, labels=labels)
    fig = px.histogram(
        df_uids.groupby(cats, axis=1).mean(),
        opacity=0.8,
        labels={"value": f"Average {value_field.title()}", "variable": "Step Quantile"},
        title=f"Average {value_field.title()}, all UIDs, by Quantile",
        width=800,
        height=600,
        template="plotly_white",
        color_discrete_sequence=px.colors.sequential.deep_r,
    ).update_layout(font_size=14, coloraxis_showscale=False)
    wandb.log({f"average_slices_{value_field}": wandb.Html(fig.to_html())})

    hit_success = df.uids.explode().value_counts().sort_index()
    fig = (
        px.bar(
            x=hit_success.index,
            y=hit_success.values,
            color=hit_success,
            labels={"x": "UID", "y": "Number of Calls"},
            title="Number of Successful Calls by UID",
            color_continuous_scale="Blues",
            height=600,
            width=800,
            template="plotly_white",
        )
        .update_traces(marker_line_color="grey")
        .update_layout(font_size=14, coloraxis_showscale=False)
    )
    wandb.log({f"uid_hit_success": wandb.Html(fig.to_html())})

    hit_total = df.all_uids.explode().value_counts().sort_index()
    fig = (
        px.bar(
            x=hit_total.index,
            y=hit_total.values,
            color=hit_total,
            labels={"x": "UID", "y": "Number of Calls"},
            title="Total Number of Calls by UID",
            color_continuous_scale="Blues",
            height=600,
            width=800,
            template="plotly_white",
        )
        .update_traces(marker_line_color="grey")
        .update_layout(font_size=14, coloraxis_showscale=False)
    )
    wandb.log({f"uid_hit_total": wandb.Html(fig.to_html())})


def run_plot(df, x, y, type="scatter", title=None, **kwargs):
    table = wandb.Table(data=df[[x, y]], columns=[x, y])

    plot_func = getattr(wandb.plot, type, None)
    if plot_func is None:
        raise ValueError(f"Unknown plot type {type!r}")

    if title is None:
        title = f"{x} vs {y}"

    wandb.log({title: plot_func(table, x, y, title=title, **kwargs)})


def run_line_series(xs, ys, keys, title):
    wandb.log({title: wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title=title)})


# TODO: Evaluate if plot3d config is ok in order to modify run_plot function to support Z (removing run_plot3d)
def run_plot3d(df, x, y, z):
    table = wandb.Table(data=df[[x, y, z]], columns=[x, y, z])
    wandb.log(
        {f"{x}_and_{y}_vs_{z}": wandb.plot.scatter(table, x, y, title=f"{x} vs {y}")}
    )


def run_features(df, feature_names, model=None, col="message"):

    # make some high level features which describe salient properties of questions such as number of words, length of question, etc.
    if "question_length" in feature_names:
        df["question_length"] = df[col].str.len()

    if "num_words" in feature_names:
        df["num_words"] = df[col].str.split().apply(len)

    if "avg_word_length" in feature_names:
        df["avg_word_length"] = df.question_length / df.num_words

    if "median_word_length" in feature_names:
        df["median_word_length"] = (
            df[col].str.split().apply(lambda x: np.median([len(w) for w in x]))
        )

    if "first_word" in feature_names:
        df["first_word"] = df[col].str.split().apply(lambda x: x[0])

    if "reward_num_tokens" in feature_names:
        df["reward_num_tokens"] = df[col].apply(
            lambda x: len(model.reward_model.tokenizer.encode(x))
        )

    return df


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
    **kwargs,
):

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

    embeddings_data = []
    pbar = tqdm.tqdm(df.message.values, unit="messages", desc="Embedding messages")
    for i, question in enumerate(pbar):

        encodings_dict = tokenizer(
            ["<|startoftext|>" + question + "<|endoftext|>"],
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        )

        with torch.no_grad():
            token_embeddings = embedding_layer(encodings_dict["input_ids"].to(device))

        embeddings_data.append(
            {
                "question": question,
                "score": df.scores[i].item(),
                "embedding": token_embeddings[0].cpu().numpy(),
                "sentence_embedding": token_embeddings[0].mean(dim=0).cpu().numpy(),
                "input_ids": encodings_dict["input_ids"].cpu().numpy(),
                "attention_mask": encodings_dict["attention_mask"].cpu().numpy(),
            }
        )

    df_embed = pd.DataFrame(embeddings_data)

    # Save to disk # TODO: Set this path from config
    df_embed.to_pickle("df_embed.pkl")

    # Prepare data for UMAP
    emblst = df_embed["embedding"].tolist()

    # embeddings (n_embeddings, max_len * embedding_dim)
    embeddings = np.stack(emblst).reshape([len(emblst), -1])

    # Grab list of scores
    scores = df_embed["score"].to_numpy()

    # Encode with umap-learn
    reducer = umap.UMAP(random_state=42)

    if scaler == "MinMax":
        scaler = MinMaxScaler()
    else:
        raise "Scaler function not defined"

    scaled_scores = scaler.fit_transform(scores.reshape(-1, 1))

    colors = plt.cm.viridis(scaled_scores)
    embedding_2d = reducer.fit_transform(embeddings)

    # Compute and save the scatter plot
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=10)
    plt.colorbar(label="Model Score")
    plt.title("UMAP Scatter Plot of Sentence Embeddings vs scores (color)")
    plt.savefig("embeddings_vs_scores_umap.png")

    # TODO: (steffen/pedro) let's chat about how to incorporate this into run_plot()
    df = pd.DataFrame()
    df[x] = embedding_2d[:, 0]
    df[y] = embedding_2d[:, 1]
    df[z] = scores

    return df


def run_preprocessing(X, y, type, **kwargs):

    if type.lower() == "pca":
        from sklearn.decomposition import PCA

        # try dimensional reduction
        pca = PCA().fit(X)

        evr_cumsum = pca.explained_variance_ratio_.cumsum()
        fig = px.line(
            y=evr_cumsum,
            labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
            template="plotly_white",
            title="Sentence Embeddings PCA Dimensional Reduction",
        )

        n_components = kwargs.get("n_components")
        if n_components:
            if 0 < n_components < 1:
                n_components = int(n_components * X.shape[1])

        threshold = kwargs.get("threshold")
        if threshold:
            n_components = np.where(evr_cumsum > threshold)[0][0]

        fig.add_vline(x=n_components, line_dash="dash", line_color="red")

        # transform sentence embeddings into n_components-dim PCA components.
        return pca.transform(X)[:, :n_components]
    else:
        raise NotImplementedError(f"Preprocessing type {type!r} not implemented.")


def run_estimator(X, y, type, **kwargs):

    from sklearn.model_selection import train_test_split

    if type == "GradientBoostingRegressor":
        from sklearn.ensemble import GradientBoostingRegressor

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        gbr = GradientBoostingRegressor(**kwargs).fit(X_train, y_train)

        preds = gbr.predict(X)
        df_model = pd.DataFrame(
            {"y": y, "y_pred": preds, "abs_error": np.abs(preds - y)}
        )
        df_model["subset"] = "train"
        df_model.loc[y_test.index, "subset"] = "test"

        print(f"Mean absolute error: {df_model.abs_error.mean():.3f}")
        print(
            f"Model training score: {gbr.score(X_train, y_train):.3f}, test score: {gbr.score(X_test, y_test):.3f}"
        )

        px.line(
            y=gbr.train_score_,
            labels={"x": "Number of Iterations", "y": "MSE Loss"},
            template="plotly_white",
            title="GradientBoostingRegressor Training Loss",
        ).show()

        if hasattr(gbr, "oob_improvement_"):
            px.line(
                y=gbr.oob_improvement_,
                labels={"x": "Number of Iterations", "y": "OOB Improvement"},
                template="plotly_white",
                title="GradientBoostingRegressor Validation Improvement",
            ).show()

        ymin, ymax = df_model.y.min(), df_model.y.max()

        fig = px.scatter(
            df_model,
            x="y",
            y="y_pred",
            color="abs_error",
            facet_col="subset",
            template="plotly_white",
            color_continuous_scale="BlueRed",
            opacity=0.3,
        )
        # add diagonal line at y=x usig fig.add_shape() and apply to all subplots
        fig.add_shape(
            x0=ymin,
            y0=ymin,
            x1=ymax,
            y1=ymax,
            line=dict(color="black", width=1),
            type="line",
            row=1,
            col=1,
        )
        fig.add_shape(
            x0=ymin,
            y0=ymin,
            x1=ymax,
            y1=ymax,
            line=dict(color="black", width=1),
            type="line",
            row=1,
            col=2,
        )
