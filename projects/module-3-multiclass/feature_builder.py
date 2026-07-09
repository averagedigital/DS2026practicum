import pandas as pd


def build_user_features(
    users,
    visits,
    ads_activity,
    surf_depth,
    primary_device,
    cloud_usage,
    feature_columns=None
):
    users = users.drop_duplicates().reset_index(drop=True)
    visits = visits.drop_duplicates().reset_index(drop=True)
    ads_activity = ads_activity.drop_duplicates().reset_index(drop=True)
    surf_depth = surf_depth.drop_duplicates().reset_index(drop=True)
    primary_device = primary_device.drop_duplicates().reset_index(drop=True)
    cloud_usage = cloud_usage.drop_duplicates().reset_index(drop=True)

    df = users.copy()

    df = df.merge(ads_activity, on='user_id', how='left')
    df = df.merge(surf_depth, on='user_id', how='left')
    df = df.merge(primary_device, on='user_id', how='left')
    df = df.merge(cloud_usage, on='user_id', how='left')

    visits_agg = (
        visits
        .groupby('user_id')
        .agg(
            total_visits=('session_id', 'count'),
            unique_categories=('website_category', 'nunique')
        )
        .reset_index()
    )

    daytime_features = (
        pd.crosstab(
            visits['user_id'],
            visits['daytime'],
            normalize='index'
        )
        .add_prefix('daytime_share_')
        .reset_index()
    )

    category_features = (
        pd.crosstab(
            visits['user_id'],
            visits['website_category'],
            normalize='index'
        )
        .add_prefix('category_share_')
        .reset_index()
    )

    visits_features = visits_agg.copy()

    visits_features = visits_features.merge(
        daytime_features,
        on='user_id',
        how='left'
    )

    visits_features = visits_features.merge(
        category_features,
        on='user_id',
        how='left'
    )

    df = df.merge(visits_features, on='user_id', how='left')

    if feature_columns is not None:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        extra_cols = [
            col for col in df.columns
            if col not in feature_columns and col != 'age_category'
        ]
        df = df.drop(columns=extra_cols)

        if 'age_category' in df.columns:
            df = df[['age_category'] + feature_columns]
        else:
            df = df[feature_columns]

    return df
