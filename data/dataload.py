import pandas as pd
from sklearn.model_selection import train_test_split

def load_df(pathname: str):
    return pd.read_csv(pathname)

def split_X_y(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return (X_train, y_train), (X_test, y_test)