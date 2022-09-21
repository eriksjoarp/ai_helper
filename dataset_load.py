import sys, os

import constants_dataset as c_d
import constants_ai_h as c_ai_h
sys.path.insert(1, c_ai_h.DIR_AI_HELPER)
sys.path.insert(1, c_ai_h.DIR_HELPER)
import helpers as h
import ml_helper as ml_h
import constants_helper as c_h

# load one or all datasets from ctu13, None loads all rows
def dataset_load_ctu13(CTU13_nr=1, NR_ROWS=None):
    ctu13_dataset = []

    DIR_DATASET_CTU13 = c_d.DIR_DATASET_CTU13
    URL_CTU13 = c_d.URL_CTU13
    FILE_EXTENSION = c_h.FILE_EXTENSION_BINETFLOW

    # update dataset if necessarry
    #h.download_file(URL_CTU13, DATASET_BASEDIR)

    # load datset into a dataframe
    DIR_DATASET_CTU13_NRX = os.path.join(DIR_DATASET_CTU13, str(CTU13_nr))
    df = ml_h.dataframes_load(DIR_DATASET_CTU13_NRX, FILE_EXTENSION, NR_ROWS, c_d.CONCATENATE_TRUE)
    return df


if __name__ == "__main__":
    ctu13_1 = dataset_load_ctu13(1)

    print(ctu13_1.head())
    print(ctu13_1.info())
    print(ctu13_1.corr())

    LABEL = 'target'
    LABEL_FROM = 'Label'
    SUBSTRING = 'botnet'

    ml_h.df_column_create_contains_text(ctu13_1, LABEL, LABEL_FROM, SUBSTRING)

    print(ctu13_1.head())

    ml_h.pandas_dataframe_describe(ctu13_1)