import pandas as pd
from pandas import DataFrame
from datetime import datetime


def extract_data(client_s3, day: datetime) -> tuple[DataFrame, DataFrame]:
    day_str = day.strftime("%Y-%m-%d")

    data = client_s3.get_object(
        Bucket="projet-datalab-group-jprat",
        Key=f"debtrank/debtrank-inputs/debtrank_inputs_snapshot_date={day_str}/nodes_info.csv",
    )["Body"]
    nodes_info = pd.read_csv(data)

    data = client_s3.get_object(
        Bucket="projet-datalab-group-jprat",
        Key=f"debtrank/debtrank-inputs/debtrank_inputs_snapshot_date={day_str}/nodes_impacts.csv",
    )["Body"]
    nodes_impacts = pd.read_csv(data)
    # data = client_s3.get_object(
    #     Bucket="projet-datalab-group-jprat",
    #     Key=f"debtrank/debtrank-inputs-dev/debtrank_inputs_snapshot_date={day_str}/nodes_impacts_without_top_100.csv",
    # )["Body"]
    # nodes_impacts = pd.read_csv(data)

    data = client_s3.get_object(
        Bucket="projet-datalab-group-jprat",
        Key=f"debtrank/debtrank-inputs/debtrank_inputs_snapshot_date={day_str}/prices_shocks.csv",
    )["Body"]
    prices_shocks = pd.read_csv(data)

    return nodes_info, nodes_impacts, prices_shocks
