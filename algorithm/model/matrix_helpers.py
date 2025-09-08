import pandas as pd
import torch
import torch.nn as nn
import numpy as np


def dataframe_to_matrix(df, values_col):
    """
    Converts a DataFrame to a matrix using one column for column names and another for values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_names_col (str): The name of the column to use for column names.
        values_col (str): The name of the column to use for values.

    Returns:
        pd.DataFrame: The resulting matrix DataFrame.
    """

    matrix_df = df.pivot_table(
        index='pre_root_id',
        columns="post_root_id",
        values=values_col,
        aggfunc="first",  # Use 'first' to handle duplicate column names
    ).fillna(
        0
    )  # Fill NaN with 0 if needed

    matrix_df.columns = matrix_df.columns.astype('str')
    matrix_df.index = matrix_df.index.astype('str')

    return matrix_df


def read_df_to_sparse_param(
    conn, rows, cols, value_col="syn_count"
):
    """return df as sparse tensor 
     Args:
        conn (pd.DataFrame): input weights (pre and postsynapse in columns)
        rows: full list of all sending neurons (codex IDs)
        cols: full list of all receiving neuorns (codex IDs)
        value_col (str): The name of the column to use for values (syn_count or nt_type)

    Returns:
        matrix: The resulting sparse tensor wrapped in nn.Parameter
    """

    matrix = dataframe_to_matrix(conn , value_col)

    matrix = matrix.reindex(index=rows, fill_value=0.0)
    matrix = matrix.reindex(columns=cols, fill_value=0.0)

    matrix = torch.tensor(np.array(matrix))

    matrix = matrix.to_sparse()

    return matrix

def new_sparse_tensor_values(v, new_values, inputs):
    indices = v.indices() 

    # Recreate sparse tensor with new values
    v_updated = torch.sparse_coo_tensor(indices.to(inputs), new_values.to(inputs), v.size())

    # If you want to overwrite v:
    v_updated
    return v_updated
