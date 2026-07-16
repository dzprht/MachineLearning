import os
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from warnings import warn


def _load_df(
    n_user_sample: int,
    random_state: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(
        "/Users/wehadgoodtimes/datasets/kuairec/KuaiRec 2.0/data/big_matrix.csv"
    )
    n_available_users = df["user_id"].nunique()

    if n_available_users < n_user_sample:
        warn(
            f"N_USERS is being decreased from {n_user_sample} to {n_available_users}",
            UserWarning,
            stacklevel=1,
        )
        n_user_sample = n_available_users

    selected_users = (
        df["user_id"]
        .drop_duplicates()
        .sample(
            n=n_user_sample,
            random_state=random_state,
        )
    )

    df = df[df["user_id"].isin(selected_users)]

    return df


def get_encoded_df(
    uf: pd.DataFrame,
    column: str,
    del_dummy: bool = True,
) -> pd.DataFrame:
    uf = uf.copy()

    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

    encoded = encoder.fit_transform(uf[[column]])
    columns = encoder.get_feature_names_out([column])
    dummy_df = pd.DataFrame(encoded, columns=columns, index=uf.index)

    uf = pd.concat([uf, dummy_df], axis=1)

    if del_dummy:
        uf.drop(columns=column, inplace=True)

    return uf


def _merge_secondary_tables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    item_categories = pd.read_csv(
        "/Users/wehadgoodtimes/datasets/kuairec/KuaiRec 2.0/data/item_categories.csv"
    )
    df = df.merge(item_categories, on="video_id", how="left")
    del item_categories

    sum_cols = [
        "show_cnt",
        "show_user_num",
        "play_cnt",
        "play_user_num",
        "play_duration",
        "complete_play_cnt",
        "complete_play_user_num",
        "valid_play_cnt",
        "valid_play_user_num",
        "long_time_play_cnt",
        "long_time_play_user_num",
        "short_time_play_cnt",
        "short_time_play_user_num",
        "like_cnt",
        "cancel_like_cnt",
        "comment_cnt",
        "comment_user_num",
        "reply_comment_cnt",
        "delete_comment_cnt",
        "comment_like_cnt",
        "follow_cnt",
        "cancel_follow_cnt",
        "share_cnt",
        "download_cnt",
        "report_cnt",
        "reduce_similar_cnt",
    ]

    mean_cols = ["play_progress", "comment_stay_duration"]

    metadata_cols = [
        "date",
        "video_type",
        "upload_dt",
        "upload_type",
        "visible_status",
        "video_duration",
        "video_width",
        "video_height",
        "music_id",
        "video_tag_id",
        "video_tag_name",
    ]

    requiered_cols = [
        "video_id",
        "author_id",
        *sum_cols,
        *mean_cols,
        *metadata_cols,
    ]

    item_daily_features = pd.read_csv(
        "/Users/wehadgoodtimes/datasets/kuairec/KuaiRec 2.0/data/item_daily_features.csv",
        usecols=requiered_cols,
    )

    item_features = item_daily_features.groupby(
        ["user_id", "video_id"],
        as_index=False,
        sort=False,
        observed=True,
    ).agg(
        **{column: "sum" for column in sum_cols},
        **{column: "mean" for column in mean_cols},
        **{column: "first" for column in metadata_cols},
    )
    del item_daily_features

    df = df.merge(item_features, how="left", on="video_id")
    del item_features
    
    features = df["feat"].fillna("[]").str.strip("[]").str.get_dummies(sep=", ")

    features = features.reindex(columns=[str(i) for i in range(31)], fill_value=0)

    features.columns = [f"feat_{i}" for i in range(31)]

    df = pd.concat([df.drop(columns="feat"), features.astype("int8")], axis=1)
    del features
    
    uf = pd.read_csv(
        "/Users/wehadgoodtimes/datasets/kuairec/KuaiRec 2.0/data/user_features.csv"
    )

    uf = get_encoded_df(uf, "user_active_degree")
    uf.drop(
        columns=[
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ],
        inplace=True,
    )

    df = df.merge(uf, how="left", on="user_id")
    del uf

    return df


def _featurize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["day_of_week"] = df["time"].dt.day_of_week
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute
    df["quarter"] = df["time"].dt.quarter

    df["is_weekend"] = (df["time"].dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = df["time"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["time"].dt.is_month_end.astype(int)

    seconds = (
        df["time"].dt.hour * 3600
        + df["time"].dt.minute * 60
        + df["time"].dt.second
        + df["time"].dt.microsecond / 1_000_000
    )
    df["time_sin"] = np.sin(2 * np.pi * seconds / 86400)
    df["time_cos"] = np.cos(2 * np.pi * seconds / 86400)

    dow = df["time"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    month = df["time"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    start_date = df["time"].min()
    df["days_from_start"] = (df["time"] - start_date).dt.seconds / 86_400

    df = df.sort_values(by=["user_id", "time"])

    df["time_since_prev"] = df.groupby("user_id")["time"].diff().dt.total_seconds()
    df["hours_since_prev"] = df["time_since_prev"] / 3600
    df["log_hours_since_prev"] = np.log1p(df["hours_since_prev"])

    df = get_encoded_df(df, "video_type")
    df = get_encoded_df(df, "upload_type")
    df = get_encoded_df(df, "visible_status")

    df["music_pop"] = df.groupby("music_id")["music_id"].transform("count")
    df["video_tag_pop"] = df.groupby("video_tag_id")["video_tag_id"].transform("count")
    df["video_tag_name_pop"] = df.groupby("video_tag_name")["video_tag_name"].transform("count")

    del df["music_id"]
    del df["video_tag_id"]
    del df["video_tag_name"]

    df["upload_dt"] = pd.to_datetime(df["upload_dt"])

    df["video_age_hours"] = (
        df["time"] - df["upload_dt"]
    ).dt.total_seconds() / 3600

    df["log_video_age"] = np.log1p(
        df["video_age_hours"].clip(lower=0)
    )

    reference_time = df["time"].max()

    df["recency_days"] = (
        reference_time - df["time"]
    ).dt.total_seconds() / 86400

    tau = 30

    df["recency_weight"] = np.exp(
        -df["recency_days"] / tau
    )

    df = df.sort_values(["user_id", "time"])

    gap_minutes = (
        df.groupby("user_id")["time"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    new_session = gap_minutes.isna() | (gap_minutes > 30)

    df["session_id"] = (
        new_session.groupby(df["user_id"])
        .cumsum()
    )

    df["position_in_session"] = (
        df.groupby(["user_id", "session_id"])
        .cumcount()
    )

    df["session_length"] = (
        df.groupby(["user_id", "session_id"])["video_id"]
        .transform("size")
    )

    return df


def make_load_df(
    n_users: int,
    random_state: int,
) -> pd.DataFrame:
    cache_dir = Path("kuairec/cache") / str(random_state)
    cache_path = cache_dir / f"dataset_{n_users}_users.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    
    df = _load_df(n_users, random_state)
    df = _merge_secondary_tables(df)
    df = _featurize_df(df)

    df.to_parquet(
        cache_path,
        index=False,
    )

    return df