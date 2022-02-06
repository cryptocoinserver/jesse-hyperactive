import csv
import logging
import os
import pathlib
import pickle
import shutil
import traceback

import click
import hyperactive
import jesse.helpers as jh
import numpy as np
import pandas as pd
import pkg_resources
import yaml
from jesse.research import backtest, get_candles


def validate_cwd() -> None:
    """
    make sure we're in a Jesse project
    """
    ls = os.listdir('.')
    is_jesse_project = 'strategies' in ls and 'storage' in ls

    if not is_jesse_project:
        print('Current directory is not a Jesse project. You must run commands from the root of a Jesse project.')
        exit()


validate_cwd()

logger = logging.getLogger()
logger.addHandler(logging.FileHandler("jesse-hyperactive.log", mode="w"))

empty_backtest_data = {'total': 0, 'total_winning_trades': None, 'total_losing_trades': None,
                       'starting_balance': None, 'finishing_balance': None, 'win_rate': None,
                       'ratio_avg_win_loss': None, 'longs_count': None, 'longs_percentage': None,
                       'shorts_percentage': None, 'shorts_count': None, 'fee': None, 'net_profit': None,
                       'net_profit_percentage': None, 'average_win': None, 'average_loss': None, 'expectancy': None,
                       'expectancy_percentage': None, 'expected_net_profit_every_100_trades': None,
                       'average_holding_period': None, 'average_winning_holding_period': None,
                       'average_losing_holding_period': None, 'gross_profit': None, 'gross_loss': None,
                       'max_drawdown': None, 'annual_return': None, 'sharpe_ratio': None, 'calmar_ratio': None,
                       'sortino_ratio': None, 'omega_ratio': None, 'serenity_index': None, 'smart_sharpe': None,
                       'smart_sortino': None, 'total_open_trades': None, 'open_pl': None, 'winning_streak': None,
                       'losing_streak': None, 'largest_losing_trade': None, 'largest_winning_trade': None,
                       'current_streak': None}


# create a Click group
@click.group()
@click.version_option(pkg_resources.get_distribution("jesse-hyperactive").version)
def cli() -> None:
    pass


@cli.command()
def create_config() -> None:
    target_dirname = pathlib.Path().resolve()
    package_dir = pathlib.Path(__file__).resolve().parent
    shutil.copy2(f'{package_dir}/hyperactive-config.yml', f'{target_dirname}/hyperactive-config.yml')


@cli.command()
def run() -> None:
    cfg = get_config()
    study_name = f"{cfg['strategy_name']}-{cfg['exchange']}-{cfg['symbol']}-{cfg['timeframe']}"

    path = f'storage/jesse-hyperactive/csv/{study_name}.csv'
    os.makedirs('./storage/jesse-hyperactive/csv', exist_ok=True)

    StrategyClass = jh.get_strategy_class(cfg['strategy_name'])
    hp_dict = StrategyClass().hyperparameters()

    search_space = get_search_space(hp_dict)

    # Later use actual search space combinations to determin n_iter / population size?
    #combinations_count = 1
    #for value in search_space.values():
        #combinations_count *= len(value)

    mem = None

    if jh.file_exists(path):
        with open(path, "r") as f:
            mem = pd.read_csv(f, sep="\t", na_values='nan')
        if not mem.empty and not click.confirm(
                f'Previous optimization results for {study_name} exists. Continue?',
                default=True,
        ):
            mem = None

    hyper = hyperactive.Hyperactive(distribution="joblib")

    if cfg['optimizer'] == "EvolutionStrategyOptimizer":
        optimizer = hyperactive.optimizers.EvolutionStrategyOptimizer(
            population=cfg[cfg['optimizer']]['population'],
            mutation_rate=cfg[cfg['optimizer']]['mutation_rate'],
            crossover_rate=cfg[cfg['optimizer']]['crossover_rate'],
            rand_rest_p=cfg[cfg['optimizer']]['rand_rest_p'],
        )

        if mem is None or len(mem) < cfg[cfg['optimizer']]['population']:
            if mem is not None and len(mem) < cfg[cfg['optimizer']]['population']:
                print('Previous optimization has too few individuals for population. Reinitialization necessary.')
            # init empty pandas dataframe
            search_data = pd.DataFrame(columns=[k for k in search_space.keys()] + ["score"] + [f'training_{k}' for k in empty_backtest_data.keys()] + [f'testing_{k}' for k in empty_backtest_data.keys()])
            with open(path, "w") as f:
                search_data.to_csv(f, sep="\t", index=False, na_rep='nan')

            hyper.add_search(objective, search_space, optimizer=optimizer,
                             initialize={"random": cfg[cfg['optimizer']]['population']},
                             n_iter=cfg['n_iter'],
                             n_jobs=cfg['n_jobs'])
        else:
            mem.drop([f'training_{k}' for k in empty_backtest_data.keys()] + [f'testing_{k}' for k in
                                                                              empty_backtest_data.keys()], 1,
                     inplace=True)
            hyper.add_search(objective, search_space, optimizer=optimizer, memory_warm_start=mem,
                             n_iter=cfg['n_iter'] -  len(mem),
                             n_jobs=cfg['n_jobs'])
    else:
        raise ValueError(f'Entered optimizer which is {cfg["optimizer"]} is not known.')

    hyper.run()


def get_config():
    cfg_file = pathlib.Path('hyperactive-config.yml')

    if not cfg_file.is_file():
        print("hyperactive-config.yml not found. Run create-config command.")
        exit()
    else:
        with open("hyperactive-config.yml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)

    return cfg


def objective(opt):
    cfg = get_config()

    try:
        training_data_metrics = backtest_function(cfg['timespan-train']['start_date'],
                                                  cfg['timespan-train']['finish_date'],
                                                  opt, cfg)
    except Exception as err:
        logger.error("".join(traceback.TracebackException.from_exception(err).format()))
        return np.nan

    if training_data_metrics is None:
        return np.nan

    if training_data_metrics['total'] <= 5:
        return np.nan

    total_effect_rate = np.log10(training_data_metrics['total']) / np.log10(cfg['optimal-total'])
    total_effect_rate = min(total_effect_rate, 1)
    ratio_config = cfg['fitness-ratio']
    if ratio_config == 'sharpe':
        ratio = training_data_metrics['sharpe_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 5)
    elif ratio_config == 'calmar':
        ratio = training_data_metrics['calmar_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 30)
    elif ratio_config == 'sortino':
        ratio = training_data_metrics['sortino_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 15)
    elif ratio_config == 'omega':
        ratio = training_data_metrics['omega_ratio']
        ratio_normalized = jh.normalize(ratio, -.5, 5)
    elif ratio_config == 'serenity':
        ratio = training_data_metrics['serenity_index']
        ratio_normalized = jh.normalize(ratio, -.5, 15)
    elif ratio_config == 'smart sharpe':
        ratio = training_data_metrics['smart_sharpe']
        ratio_normalized = jh.normalize(ratio, -.5, 5)
    elif ratio_config == 'smart sortino':
        ratio = training_data_metrics['smart_sortino']
        ratio_normalized = jh.normalize(ratio, -.5, 15)
    else:
        raise ValueError(
            f'The entered ratio configuration `{ratio_config}` for the optimization is unknown. Choose between sharpe, calmar, sortino, serenity, smart shapre, smart sortino and omega.')
    if ratio < 0:
        return np.nan

    score = total_effect_rate * ratio_normalized

    try:
        testing_data_metrics = backtest_function(cfg['timespan-testing']['start_date'],
                                                 cfg['timespan-testing']['finish_date'], opt, cfg)
    except Exception as err:
        logger.error("".join(traceback.TracebackException.from_exception(err).format()))
        return np.nan

    if testing_data_metrics is None:
        return np.nan

    # you can access the entire dictionary from "para"
    parameter_dict = opt.para_dict

    # save the score in the copy of the dictionary
    parameter_dict["score"] = score

    for key, value in training_data_metrics.items():
        parameter_dict[f'training_{key}'] = value

    for key, value in testing_data_metrics.items():
        parameter_dict[f'testing_{key}'] = value

    study_name = f"{cfg['strategy_name']}-{cfg['exchange']}-{cfg['symbol']}-{cfg['timeframe']}"

    path = f'storage/jesse-hyperactive/csv/{study_name}.csv'

    # append parameter dictionary to csv
    with open(path, "a") as f:
        writer = csv.writer(f, delimiter='\t')
        fields = parameter_dict.values()
        writer.writerow(fields)

    return score


def get_search_space(strategy_hps):
    hp = {}
    for st_hp in strategy_hps:
        if st_hp['type'] is int:
            if 'step' not in st_hp:
                st_hp['step'] = 1
            hp[st_hp['name']] = list(range(st_hp['min'], st_hp['max'] + st_hp['step'], st_hp['step']))
        elif st_hp['type'] is float:
            if 'step' not in st_hp:
                st_hp['step'] = 0.1
            decs = str(st_hp['step'])[::-1].find('.')
            hp[st_hp['name']] = list(
                np.trunc(np.arange(st_hp['min'], st_hp['max'] + st_hp['step'], st_hp['step']) * 10 ** decs) / (
                        10 ** decs))
        elif st_hp['type'] is bool:
            hp[st_hp['name']] = [True, False]
        else:
            raise TypeError('Only int, bool and float types are implemented')
    return hp


def get_candles_with_cache(exchange: str, symbol: str, start_date: str, finish_date: str) -> np.ndarray:
    path = pathlib.Path('storage/jesse-hyperactive')
    path.mkdir(parents=True, exist_ok=True)

    cache_file_name = f"{exchange}-{symbol}-1m-{start_date}-{finish_date}.pickle"
    cache_file = pathlib.Path(f'storage/jesse-hyperactive/{cache_file_name}')

    if cache_file.is_file():
        with open(f'storage/jesse-hyperactive/{cache_file_name}', 'rb') as handle:
            candles = pickle.load(handle)
    else:
        candles = get_candles(exchange, symbol, '1m', start_date, finish_date)
        with open(f'storage/jesse-hyperactive/{cache_file_name}', 'wb') as handle:
            pickle.dump(candles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return candles


def backtest_function(start_date, finish_date, hp, cfg):
    candles = {}
    extra_routes = []
    if len(cfg['extra_routes']) != 0:
        for extra_route in cfg['extra_routes'].items():
            extra_route = extra_route[1]
            candles[jh.key(extra_route['exchange'], extra_route['symbol'])] = {
                'exchange': extra_route['exchange'],
                'symbol': extra_route['symbol'],
                'candles': get_candles_with_cache(
                    extra_route['exchange'],
                    extra_route['symbol'],
                    start_date,
                    finish_date,
                ),
            }
            extra_routes.append({'exchange': extra_route['exchange'], 'symbol': extra_route['symbol'],
                                 'timeframe': extra_route['timeframe']})
    candles[jh.key(cfg['exchange'], cfg['symbol'])] = {
        'exchange': cfg['exchange'],
        'symbol': cfg['symbol'],
        'candles': get_candles_with_cache(
            cfg['exchange'],
            cfg['symbol'],
            start_date,
            finish_date,
        ),
    }

    route = [{'exchange': cfg['exchange'], 'strategy': cfg['strategy_name'], 'symbol': cfg['symbol'],
              'timeframe': cfg['timeframe']}]

    config = {
        'starting_balance': cfg['starting_balance'],
        'fee': cfg['fee'],
        'futures_leverage': cfg['futures_leverage'],
        'futures_leverage_mode': cfg['futures_leverage_mode'],
        'exchange': cfg['exchange'],
        'settlement_currency': cfg['settlement_currency'],
        'warm_up_candles': cfg['warm_up_candles']
    }

    backtest_data = backtest(config, route, extra_routes, candles, hp)

    if backtest_data['total'] == 0:
        backtest_data = empty_backtest_data

    return backtest_data
