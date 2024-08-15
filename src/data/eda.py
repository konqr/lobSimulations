import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
import time
import pickle
import gc
import itertools

sys.path.append("/home/konajain/code/lobSimulations")
from src.backup.hawkes import dataLoader


def nanmed(data):
    return pd.Series({
        'q_LO': np.nanmedian(data['q_LO']),
        'q_MO': np.nanmedian(data['q_MO']),
        'eta_is': np.nanmedian(data['eta_is'])
    })


def process_data(ric, eda_flags):
    samplingTime = 60

    if eda_flags.get('edasprea'):
        process_spread(ric, samplingTime)

    if eda_flags.get('edashape'):
        process_shape(ric)

    if eda_flags.get('edasparse'):
        process_sparse(ric)

    if eda_flags.get('edarest'):
        process_rest(ric)


def process_spread(ric, samplingTime):
    spreads = []
    for date in pd.date_range(dt.date(2019, 1, 2), dt.date(2019, 12, 31)):
        loader = dataLoader.Loader(ric, date, date, nlevels=1, dataPath="/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
        data = loader.load()
        if data:
            data = data[0]
            data['spread'] = data['Ask Price 1'] - data['Bid Price 1']
            data['timeDiff'] = data['Time'].diff()
            data['spreadTwa'] = data['spread'] * data['timeDiff']
            data['id'] = data['Time'].apply(lambda x: int((x - 34200) // samplingTime))
            twaspread = (data[['spreadTwa', 'id']].groupby('id').sum().values) / samplingTime
            spreads.append(twaspread)

    plot_spreads(spreads, ric, samplingTime)
    save_spreads(spreads, ric)


def plot_spreads(spreads, ric, samplingTime):
    plt.figure()
    for s in spreads:
        plt.plot(100 * s, alpha=0.01, color="steelblue")
    avg = np.average(100 * np.array([s for s in spreads if len(s) == 390]), axis=0)
    plt.plot(avg, color="r", label="average")
    plt.xlabel("Time")
    plt.ylabel("Spread in ticks")
    plt.legend()
    plt.title(f"Spread by TOD - {ric}")
    plt.xticks(
        ticks=np.arange(0, 23400 // samplingTime, 1800 // samplingTime),
        labels=[time.strftime('%H:%M:%S', time.gmtime(x)) for x in 9.5 * 3600 + samplingTime * np.arange(0, 23400 // samplingTime, 1800 // samplingTime)],
        rotation=20
    )
    plt.savefig(f"/SAN/fca/Konark_PhD_Experiments/smallTick/{ric}_EDA_SpreadTOD.png")
    plt.show()


def save_spreads(spreads, ric):
    avg = np.average(100 * np.array([s for s in spreads if len(s) == 390]), axis=0)
    with open(f"/SAN/fca/Konark_PhD_Experiments/smallTick/{ric}_EDA_SpreadTOD", "wb") as f:
        pickle.dump(avg, f)

    plt.figure()
    hist_spread(avg, ric, 'mediumpurple', 'Spread density (log)', 'EDA_SpreadDistriLog')
    plt.figure()
    hist_spread(avg, ric, 'lightcoral', 'Spread density', 'EDA_SpreadDistri')


def hist_spread(avg, ric, color, title, save_name):
    plt.hist(np.ravel(100 * np.array([s for s in avg if len(s) == 390])), bins=1000, density=True, color=color)
    plt.xlabel("Spread in ticks")
    plt.yscale("log")
    plt.text(
        plt.xlim()[1] * 0.75,
        plt.ylim()[1] * 0.8,
        f"Mean = {np.round(np.average(avg), 2)}",
        bbox=dict(alpha=0.5)
    )
    plt.title(f"{title} : {ric}")
    plt.savefig(f"/SAN/fca/Konark_PhD_Experiments/smallTick/{ric}_{save_name}.png")


def process_shape(ric):
    final_res, final_res_far = {}, {}
    for date in pd.date_range(dt.date(2019, 1, 2), dt.date(2019, 12, 31)):
        if date == dt.date(2019, 1, 9):
            continue
        loader = dataLoader.Loader(ric, date, date, nlevels=10, dataPath="/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
        data = loader.load()
        if not data:
            continue
        data = data[0]
        data['timeDiff'] = data['Time'].diff()
        data['total_v_a'] = data[[f"Ask Size {i}" for i in range(1, 11)]].sum(axis=1)
        data['total_v_b'] = data[[f"Bid Size {i}" for i in range(1, 11)]].sum(axis=1)
        data['mid'] = (data['Ask Price 1'] + data['Bid Price 1']) * 0.5

        process_levels(data, final_res, final_res_far)

    plot_shape(final_res, ric, "Shape of LOB")
    save_shape(final_res, ric, "EDA_Shape")
    save_shape(final_res_far, ric, "EDA_ShapeFT")
    plot_shape_percentiles(final_res, ric)


def process_levels(data, final_res, final_res_far):
    for i in range(10, 0, -1):
        data[f"Ask Size {i}"] /= data['total_v_a']
        data[f"Bid Size {i}"] /= data['total_v_b']
        data[f"Ask Price1 {i}"] = np.round(200 * (data[f"Ask Price {i}"] - data['mid'])).astype(int)
        data[f"Bid Price1 {i}"] = np.round(200 * (data['mid'] - data[f"Bid Price {i}"])).astype(int)
        data[f"Ask Price2 {i}"] = np.round(200 * (data[f"Ask Price {i}"] - data["Bid Price 1"])).astype(int)
        data[f"Bid Price2 {i}"] = np.round(200 * (data['Ask Price 1'] - data[f"Bid Price {i}"])).astype(int)
        update_final_res(data, final_res, final_res_far, i)


def update_final_res(data, final_res, final_res_far, i):
    data['tmp'] = data[f'Ask Size {i}'] * data['timeDiff']
    data_dict = (data.groupby(f"Ask Price1 {i}")['tmp'].sum() / 23400).to_dict()
    data_dict_far = (data.groupby(f"Ask Price2 {i}")['tmp'].sum() / 23400).to_dict()
    merge_dicts(data_dict, final_res)
    merge_dicts(data_dict_far, final_res_far)
    data['tmp'] = data[f'Bid Size {i}'] * data['timeDiff']
    data_dict = (data.groupby(f"Bid Price1 {i}")['tmp'].sum() / 23400).to_dict()
    data_dict_far = (data.groupby(f"Bid Price2 {i}")['tmp'].sum() / 23400).to_dict()
    merge_dicts(data_dict, final_res)
    merge_dicts(data_dict_far, final_res_far)


def merge_dicts(source, target):
    for k, v in source.items():
        target[k] = target.get(k, []) + [v]


def plot_shape(final_res, ric, title):
    plt.figure()
    for i in range(len(final_res[list(final_res.keys())[0]])):
        x, y = [], []
        for k in range(1, 250):
            if (final_res.get(k, None) is not None) and (len(final_res.get(k, [])) == len(final_res[list(final_res.keys())[0]])):
                x.append(k)
                y.append(final_res[k][i])
        x = np.array(x)
        if ric in ['SIRI', 'BAC', 'INTC', 'CSCO', 'ORCL', 'MSFT']:
            x, y = refine_shape(x, y)
        plt.plot(x / 2, y, alpha=0.1, color="steelblue")
    plt.plot(x / 2, [np.average(final_res[k]) for k in x], color="r")
    plt.xlabel("Depth from mid (in ticks)")
    plt.ylabel("Volume ratio")
    plt.title(f"{title} - {ric}")
    plt.legend()
    plt.savefig(f"/SAN/fca/Konark_PhD_Experiments/smallTick/{ric}_{title.replace(' ', '_')}.png")


def refine_shape(x, y):
    x = x[0:-1:2]
    x = [i for i in x if i <= 20]
    y = [sum(y[i*2:i*2+2]) for i in range(len(x))]
    return x, y


def save_shape(final_res, ric, save_name):
    avg_shape = [np.average(final_res[k]) for k in np.arange(1, 250)]
    with open(f"/SAN/fca/Konark_PhD_Experiments/smallTick/{ric}_{save_name}", "wb") as f:
        pickle.dump(avg_shape, f)


def plot_shape_percentiles(final_res, ric):
    percentiles = np.percentile([v for k, v in final_res.items()], [25, 50, 75], axis=1)
    plt.figure()
    plt.plot(np.arange(1, 250), percentiles[0], color="g", linestyle="--", label="25th percentile")
    plt.plot(np.arange(1, 250), percentiles[1], color="b", linestyle="--", label="50th percentile")
    plt.plot(np.arange(1, 250), percentiles[2], color="r", linestyle="--", label="75th percentile")
    plt.legend()
    plt.xlabel("Depth from mid (in ticks)")
    plt.ylabel("Volume ratio")
    plt.title(f"Shape Percentiles - {ric}")
    plt.savefig(f"/SAN/fca/Konark_PhD_Experiments/smallTick/{ric}_ShapePercentiles.png")
    plt.show()


def process_sparse(ric):
    results = []
    for date in pd.date_range(dt.date(2019, 1, 2), dt.date(2019, 12, 31)):
        loader = dataLoader.Loader(ric, date, date, nlevels=10, dataPath="/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
        data = loader.load()
        if not data:
            continue
        data = data[0]
        data['eta_is'] = data['MidPrice'] / (data['Ask Price 1'] + data['Bid Price 1'])
        data['q_LO'] = (data['Ask Size 1'] - data['Bid Size 1']) / (data['Ask Size 1'] + data['Bid Size 1'])
        data['q_MO'] = data['MidPrice'] - data['MidPrice'].shift(1)
        results.append(data[['q_LO', 'q_MO', 'eta_is']].resample('1H', on='Time').apply(nanmed))

    save_sparse(results, ric)


def save_sparse(results, ric):
    with open(f"/SAN/fca/Konark_PhD_Experiments/smallTick/{ric}_Sparse", "wb") as f:
        pickle.dump(results, f)


def process_rest(ric):
    results = []
    for date in pd.date_range(dt.date(2019, 1, 2), dt.date(2019, 12, 31)):
        loader = dataLoader.Loader(ric, date, date, nlevels=10, dataPath="/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
        data = loader.load()
        if not data:
            continue
        data = data[0]
        data['mid'] = (data['Ask Price 1'] + data['Bid Price 1']) * 0.5
        data['timeDiff'] = data['Time'].diff()
        results.append(data[['mid', 'timeDiff']])

    save_rest(results, ric)


def save_rest(results, ric):
    with open(f"/SAN/fca/Konark_PhD_Experiments/smallTick/{ric}_Rest", "wb") as f:
        pickle.dump(results, f)

import numpy as np
import pandas as pd
import pickle
import datetime as dt
from dataLoader import Loader


def process_qd(ric):
    wts, chgs = [], []
    for date in pd.date_range(dt.date(2019, 1, 2), dt.date(2019, 12, 31)):
        if date == dt.date(2019, 1, 9):  # Skip specific date
            continue

        data = load_data(ric, date)
        if data is None:
            continue

        data = calculate_price_changes(data)
        data_trade = data[data['Type'] == 4]
        wts.extend(data_trade.timeDiff.values)
        chgs.extend((data_trade['Ask Price change'] + data_trade['Bid Price change']).values)

    save_results(f"{ric}_EDA_MOQDs", (wts, chgs))


def process_shape_maxima(ric):
    dict_res = {}
    for date in pd.date_range(dt.date(2019, 1, 2), dt.date(2019, 12, 31)):
        if date == dt.date(2019, 1, 9):  # Skip specific date
            continue

        data = load_data(ric, date)
        if data is None:
            continue

        data = calculate_depth_prices(data)
        maxdepths = calculate_max_depths(data)
        counts = np.unique(maxdepths, return_counts=True)

        for k, v in zip(counts[0], counts[1]):
            dict_res[k] = dict_res.get(k, 0) + v

    save_results(f"{ric}_EDA_shapeMaxima", dict_res)


def process_shape_sparsity(ric):
    with open(f"/SAN/fca/Konark_PhD_Experiments/smallTick/{ric}_EDA_Shape", "rb") as f:
        data = pickle.load(f)

    avg_shape = calculate_avg_shape(data, ric)
    wass_distances_mean, wass_distances_var = [], []

    for date in pd.date_range(dt.date(2019, 1, 2), dt.date(2019, 12, 31)):
        if date == dt.date(2019, 1, 9):  # Skip specific date
            continue

        data = load_data(ric, date)
        if data is None:
            continue

        wass_mean, wass_var = calculate_wasserstein_distances(data, avg_shape)
        wass_distances_mean.extend(wass_mean)
        wass_distances_var.extend(wass_var)

    save_results(f"{ric}_EDA_shapeWasserstein", (wass_distances_mean, wass_distances_var))


def process_leverage(ric, one_sided_top=False):
    cond_counts, uncond_counts = [], []

    for date in pd.date_range(dt.date(2019, 1, 2), dt.date(2019, 12, 31)):
        if date == dt.date(2019, 1, 9):  # Skip specific date
            continue

        data = load_data(ric, date)
        if data is None:
            continue

        if one_sided_top:
            data = process_one_sided_top(data)
        else:
            data = process_depth_events(data)

        cond_counts, uncond_counts = update_counts(data, cond_counts, uncond_counts)

    leverage = calculate_leverage(cond_counts, uncond_counts)
    suffix = "OneSidedTop" if one_sided_top else "leverage"
    save_results(f"{ric}_EDA_{suffix}", leverage)


def load_data(ric, date):
    loader = Loader(ric, date, date, nlevels=10, dataPath="/SAN/fca/Konark_PhD_Experiments/extracted/GOOG/")
    data = loader.load()
    if data:
        return data[0]
    return None


def calculate_price_changes(data):
    data['timeDiff'] = data['Time'].diff()
    data['Ask Price change'] = data['Ask Price 1'] - data['Ask Price 1'].shift(1)
    data['Bid Price change'] = data['Bid Price 1'] - data['Bid Price 1'].shift(1)
    return data


def calculate_depth_prices(data):
    data['mid'] = (data['Ask Price 1'] + data['Bid Price 1']) * 0.5
    for i in range(10, 0, -1):
        data[f"Ask Price1 {i}"] = np.round(200 * (data[f"Ask Price {i}"] - data['mid'])).astype(int)
        data[f"Bid Price1 {i}"] = np.round(200 * (data['mid'] - data[f"Bid Price {i}"])).astype(int)
    return data


def calculate_max_depths(data):
    maxdepths = []
    for side in ["Ask", "Bid"]:
        maxidxs = np.argmax(data[[f"{side} Size {i}" for i in range(1, 11)]].values, axis=1)
        depths = data[[f"{side} Price1 {i}" for i in range(1, 11)]].values[np.arange(len(data)), maxidxs]
        maxdepths.extend(depths)
    return maxdepths


def calculate_avg_shape(data, ric):
    x = [k for k in range(1, 250) if data.get(k) and len(data[k]) == len(data[list(data.keys())[0]])]
    if ric in ['SIRI', 'BAC', 'INTC', 'CSCO', 'ORCL', 'MSFT']:
        x = [i for i in x[::2] if i <= 20]
    elif ric in ['AAPL', 'ABBV', 'PM', 'IBM']:
        x = [i for i in x[::2] if i <= 20]

    avg_shape_lvls = np.array(x)
    avg_shape_vols = np.array([np.average(data[k]) for k in avg_shape_lvls]) / np.sum(
        [np.average(data[k]) for k in avg_shape_lvls])
    avg_shape = np.zeros(int(max(avg_shape_lvls)) + 1)
    avg_shape[avg_shape_lvls.astype(int)] = avg_shape_vols
    return avg_shape


def calculate_wasserstein_distances(data, avg_shape):
    wass_distances_mean, wass_distances_var = [], []
    data = calculate_depth_prices(data)
    data['total v_a'] = data[[f"Ask Size {i}" for i in range(1, 11)]].sum(axis=1)
    data['total v_b'] = data[[f"Bid Size {i}" for i in range(1, 11)]].sum(axis=1)

    for side in ["Ask", "Bid"]:
        instt_lvls = data[[f"{side} Price1 {i}" for i in range(1, 11)]].values
        instt_vols = data[[f"{side} Size {i}" for i in range(1, 11)]].values / data[f"total v_{side[0].lower()}"]

        wass_distances = calculate_individual_wasserstein(instt_lvls, instt_vols, avg_shape)
        mean_wass_distances = np.nansum(data['timeDiff'].values * np.array(wass_distances)) / 23400
        var_wass_distances = np.average(
            (np.array(wass_distances) - mean_wass_distances) ** 2, weights=np.nan_to_num(data['timeDiff'].values)
        )
        wass_distances_mean.append(mean_wass_distances)
        wass_distances_var.append(var_wass_distances)

    return wass_distances_mean, wass_distances_var


def calculate_individual_wasserstein(instt_lvls, instt_vols, avg_shape):
    wass_distances = []
    for instt_lvl, instt_vol in zip(instt_lvls, instt_vols):
        instt_shape = np.zeros(int(max(avg_shape)) + 1)
        valid = instt_lvl <= max(avg_shape)
        instt_shape[instt_lvl[valid]] = instt_vol[valid]
        wass_distances.append(np.abs(avg_shape - instt_shape).sum())
    return wass_distances


def process_depth_events(data):
    data = data[data['Type'] < 5]
    data = data[data['Type'] != 2]
    data['mid'] = (data['Ask Price 1'] + data['Bid Price 1']) * 0.5
    data['midPrev'] = data['mid'].shift(1).fillna(0)
    data['depth'] = np.round(100 * (data.Price / 10000 - data['midPrev']), 2)
    data['depthAbs'] = data['depth'].abs()
    data['TypeDepth'] = data['Type'].astype(str) + data['depth'].astype(int).astype(str)
    data['TypeDepth_1'] = data['TypeDepth'].shift(1)
    return data


def process_one_sided_top(data):
    data_orig = data.copy()
    data_list = []

    for d, side in zip([-1, 1], ['Ask', 'Bid']):
        data = data_orig[data_orig['TradeDirection'] == d]
        data['topPrev'] = data[f'{side} Size 1'].shift(1).fillna(0)
        data = data[(data['Type'] == 4) & (data['TradeVolume'] >= 0.1 * data['topPrev'])]

        data['mid'] = (data['Ask Price 1'] + data['Bid Price 1']) * 0.5
        data['midPrev'] = data['mid'].shift(1).fillna(0)
        data['depth'] = np.round(100 * (data.Price / 10000 - data['midPrev']), 2)
        data['depthAbs'] = data['depth'].abs()
        data['TypeDepth'] = data['Type'].astype(str) + data['depth'].astype(int).astype(str)
        data['TypeDepth_1'] = data['TypeDepth'].shift(1)
        data_list.append(data)

    return pd.concat(data_list)


def update_counts(data, cond_counts, uncond_counts):
    cond_counts.append(data['TypeDepth'].value_counts())
    uncond_counts.append(data['TypeDepth_1'].value_counts())
    return cond_counts, uncond_counts


def calculate_leverage(cond_counts, uncond_counts):
    leverage = pd.concat(cond_counts, axis=1).sum(axis=1) / pd.concat(uncond_counts, axis=1).sum(axis=1)
    leverage = leverage.replace([np.inf, -np.inf], np.nan).dropna()
    return leverage


def save_results(filename, data):
    with open(f"/SAN/fca/Konark_PhD_Experiments/smallTick/{filename}", "wb") as f:
        pickle.dump(data, f)

