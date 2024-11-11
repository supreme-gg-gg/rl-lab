import matplotlib.pyplot as plt
import matplotlib
import torch
import pandas as pd, numpy as np

def get_data(stock, scale=True):

    '''
    Obtains and preprocesses the data from the file path.
    The file path must be relative to the current working directory.
    Performs regularisation if scale is set to True.
    Computes return as well as technical indicators
    that provides a total of 10 feature to the agent.
    '''
    filepath = f"/Users/supremegg/Documents/GitHub/financial-ml/data/{stock}.csv"
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    df.Volume = df.Volume.replace(0,1)
    df.drop(columns=["Close"], inplace=True)
    df['Return'] = df["Adj Close"].pct_change()

    # these functions append to df directly
    get_sma(df)
    get_ema(df)
    get_macd(df)
    get_bias(df)
    get_vvr(df)

    R = df.Return
    if scale:
        mean = df.mean(axis=0)
        std = df.std(axis=0)
        df = (df - mean) / std
    df['Return'] = R # Return is not scaled

    # The first 15 rows are removed as they contain NaN values
    # due to the computation of technical indicators

    return df[15:]

def get_sma(df, window=15):

    '''
    SMA = (Sum of prices for the last N days) / N
    '''

    df["SMA"] = df["Adj Close"].rolling(window).mean().shift()

def get_ema(df, window=5):

    # EMA = (Price(t) * k) + (EMA(y) * (1 - k))

    df["EMA"] = df["Adj Close"].ewm(window).mean().shift()

def get_macd(df, slow=26, fast=12):

    # MACD = EMA(fast) - EMA(slow)

    df["MACD"] = df["Adj Close"].ewm(fast).mean().shift() - df["Adj Close"].ewm(slow).mean().shift()

def get_bias(df):

    # BIAS = (Price - SMA) / SMA
    
    df["BIAS"] = (df["Adj Close"] - df["SMA"]) / df["SMA"]

def get_vvr(df, window=14):

    # VVR = TR / ATR where TR = max(High - Low, High - Close, Close - Low), ATR = average TR over N days
    
    df["TTR"] = np.maximum((df["High"] - df["Low"]), np.abs(df["High"] - df["Adj Close"].shift()), np.abs(df["Low"] - df["Adj Close"].shift()))
    df["ATR"] = df["TTR"].rolling(window).mean()

    df["VVR"] = df["TTR"] / df["ATR"]

    df.drop(columns=["TTR", "ATR"], inplace=True)

def plot_durations(episode_durations, show_result=False):
    
    '''
    Plot the duration of each episode, 
    as well as the mean duration of the last 100 episodes.

    Args:
    - episode_durations (list): list of durations of each episode
    - show_result (bool): whether to show the result or not

    Returns:
    - None
    '''
    
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
    
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())