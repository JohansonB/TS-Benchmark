import time

from tbats import BATS
from tbats import TBATS
import numpy as np
import argparse

# required on windows for multi-processing,
# see https://docs.python.org/2/library/multiprocessing.html#windows

def rescale(df):
    minimum = np.min(df)
    df = (df - minimum)
    maximum = np.max(df)
    df = df / maximum

    return minimum, maximum, df


def inverse_rescale(df, minimum, maximum):
    return df * maximum + minimum


def rescale_row_wise(df):
    if df.ndim == 1:
        return rescale(df)
    else:
        minimum = []
        maximum = []
        for i in range(len(df)):
            a, b, c = rescale(df[i])
            df[i] = c;
            minimum.append(a)
            maximum.append(b)
        return minimum, maximum, df


def inverse_rescale_row_wise(df, minimum, maximum):
    if df.ndim == 1:
        return inverse_rescale(df, minimum, maximum)
    else:
        for i in range(len(df)):
            df[i] = inverse_rescale(df[i], minimum[i], maximum[i])
        return df


def parse_args():
    parser = argparse.ArgumentParser()
    # ALL
    parser.add_argument('--output_len', type=int)
    parser.add_argument('--input_chunk', type=int, default=None)
    parser.add_argument('--normalize', type=str, default="True")
    parser.add_argument('--algo', type=str)
    parser.add_argument('--periods_TBATS', nargs="+", type=int, default=None)
    parser.add_argument('--trend_TBATS', type=str, default="False")
    # Dataset setting
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)

    # parse the arguments
    args = parser.parse_args()

    return args




def str_to_bool(str):
    if str == "False":
        return False
    if str == "True":
        return True
def main():
    from sktime.datasets import load_airline
    runTime, arr = TBATS_exe()
    args = parse_args()

    with open('Scripts/Python/Output/' + args.algo + '_Out.txt', 'w') as the_file:

        the_file.write(str(runTime))
        the_file.write('\n')
        if arr.ndim > 1:
            length = arr.shape[1]
        else:
            length = len(arr)
        count = 1
        for x in arr.flatten():
            the_file.write(str(x))
            the_file.write(' ')
            if (count == length):
                count = 1
                the_file.write("\n")
            else:
                count = count + 1

def TBATS_exe():
    args = parse_args()

    args.normalize = str_to_bool(args.normalize)
    args.trend_TBATS = str_to_bool(args.trend_TBATS)
    modelName = args.algo
    train_path = args.train_path
    test_path = args.test_path

    test = np.genfromtxt(test_path, delimiter=",");
    forecast_length = len(test)



    #if there is a -1 in the periods list this indicates that the periods shell be set o NONE
    if args.periods_TBATS is not None:
        if -1 in args.periods_TBATS:
            args.periods_TBATS = None

    if args.normalize:
        minimum, maximum, s = rescale_row_wise(np.genfromtxt(train_path, delimiter=","))
    else:
        s = np.genfromtxt(train_path, delimiter=",")

    df = s


    if modelName == "BATS":
        forecaster = BATS(seasonal_periods=args.periods_TBATS,use_trend=args.trend_TBATS,n_jobs=1)
    else:
        forecaster = TBATS(seasonal_periods=args.periods_TBATS,use_trend=args.trend_TBATS,n_jobs=1)

    start = time.perf_counter()
    model = forecaster.fit(df)
    end = time.perf_counter()
    runTime = end - start

    if args.normalize:
        y_pred = inverse_rescale_row_wise(model.forecast(steps=forecast_length),
                                          minimum, maximum)
    else:
        y_pred = model.forecast(steps=forecast_length);
    return runTime, y_pred


if __name__ == '__main__':
    main()

