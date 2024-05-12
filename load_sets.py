from sklearn.model_selection import train_test_split


def load_train_test_sets(df, target_column='Measured_leaf_area', by_year=False):
    '''
    A function to load train and test data sets based on a target column and year.
    :param df: dataframe containing reconstruction results for one parameter name and value
    :param target_column: i guess, leaf area. might be changed to whatever is needed.
    :param by_year: if true, allows to train models on year 2023 and test on year 2024.
    :return: 4 dataframes containing train and test data sets.
    '''
    df.loc[:, 'Year'] = ('20' + df['File_name'].str[:6].str[-2:]).astype(int)
    df_train = df[df['Year'] == 2023]
    df_test = df[df['Year'] == 2024]
    if by_year:
        X_train = df_train.drop(columns=[target_column, 'File_name', 'parameter_type', 'parameter_value', 'Year'],
                                inplace=False, axis=1)
        y_train = df_train[target_column]
        X_test = df_test.drop(columns=[target_column, 'File_name', 'parameter_type', 'parameter_value', 'Year'],
                              inplace=False, axis=1)
        y_test = df_test[target_column]
    else:
        df.drop(columns=['File_name', 'parameter_type', 'parameter_value', 'Year'],
                      inplace=True, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(df.drop(target_column, axis=1), df[target_column],
                                                            test_size=0.3, random_state=42)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # print(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test