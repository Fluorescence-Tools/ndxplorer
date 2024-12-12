from typing import Dict, List
import sys
import json
import pandas as pd
import numpy as np
from . data_source import DataSource
if sys.version_info.major > 2:
    import pathlib
else:
    import pathlib2 as pathlib


def read_paris_analysis(
        base_path: str = "./test/mfd/burstwise_All 0.2500#30",
        skip_nth_row: int = 2
) -> DataSource:
    base_path = pathlib.Path(base_path)
    path = base_path / "bur"
    filenames = path.glob("*.bur")
    df_files = list()
    for filename in filenames:
        dfs = list()
        df = pd.read_csv(filename, sep="\t")
        df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
        dfs.append(df)
        fn_head = filename.name.split(".")[0]
        for ending in ["bg4", "br4"]:
            fn_name = fn_head + "." + ending
            fn = base_path / ending / fn_name
            df = pd.read_csv(fn, sep="\t")
            df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
            dfs.append(df)
        b = pd.concat(dfs, axis=1)
        df_files.append(b[b.index % skip_nth_row != 0])
    df = pd.concat(df_files)
    data_source = DataSource()
    data_source.data = df
    return data_source


def read_csv_sampling(filenames, sep='\t'):
    # type: (List[str])->(DataSource)
    with open(filenames[0], "r") as fp:
        l = fp.readline()
        pn = l.split("\t")
    values = list()
    for filename in filenames:
        df = pd.read_csv(filename, sep=sep)
        values.append(df)
    data = pd.concat(values)
    return DataSource(
        data=data,
        parameter_names=pn
    )


def read_csv(filenames):
    df_files = list()
    for filename in filenames:
        df = pd.read_csv(filename, sep="\t")
        df_files.append(df)
    dfs = pd.concat(df_files)
    dfn = dfs.select_dtypes(['number'])
    ds = DataSource()
    ds.data = dfn
    return ds


if __name__ == "__main__":
    equation_json_fn = pathlib.Path(__file__).parent / "mfd.equations.yaml"
    constants_json_fn = pathlib.Path(__file__).parent / "mfd.constants.json"
    with open(constants_json_fn, "r") as fp:
        constants = json.load(fp)
    df = read_paris_analysis()
