import polars as pl
import polars.selectors as cs
from yfh.functions import availability_probability, get_next_round_expectation


def load_data(path, sheet_name):
    return (
        pl.read_excel(path, sheet_name=sheet_name)
        .select(
            pl.col('NAME').alias('player'),
            pl.col('RK').alias('rk'),
            pl.col('POS').alias('pos'),
            pl.col('TEAM').alias('team'),
            pl.col('FP').alias('pts'),
            pl.col('VORP').alias('vorp'),
            pl.col('ADP').cast(pl.Float64, strict=False).alias('adp'),
        )
        .with_columns(pl.col('adp').fill_null(pl.max('adp')))
        .with_columns(
            pl.col('adp').mul(2).add(pl.col('rk')).truediv(3).alias('tru')
        )
        .with_columns(
            (abs(pl.col('adp') - pl.col('tru'))).clip(0, 50).alias('adp_diff')
        )
        .with_columns(
            (0.25 * pl.col('tru') + 0.25 * pl.col('adp_diff')).alias('sd')
        )
        .select('player', 'rk', 'pos', 'team', 'pts', 'vorp', pl.col('tru').alias('adp'), pl.col('sd').alias('adp_sd'))
        .with_row_index('id')
    )


def process_probs(df, pick_1, pick_2, pick_3):
    return (
        df
        .with_columns(
            pl.struct(['adp', 'adp_sd', pl.lit(pick_1).alias('pick')])
            .map_elements(availability_probability, return_dtype=pl.Float64)
            .alias("prob_1"),
            pl.struct(['adp', 'adp_sd', pl.lit(pick_2).alias('pick')])
            .map_elements(availability_probability, return_dtype=pl.Float64)
            .alias("prob_2"),
            pl.struct(['adp', 'adp_sd', pl.lit(pick_3).alias('pick')])
            .map_elements(availability_probability, return_dtype=pl.Float64)
            .alias("prob_3"),
        )
    )


def process_dropoffs(df):
    return (
        df
        .with_columns([
            (pl.col('vorp') - get_next_round_expectation(df, pos, pick)).alias(f"dropoff_{pos}_{pick}")
            for pos in ['C', 'LW', 'RW', 'D', 'G']
            for pick in [1, 2, 3]]
        )
        .with_columns([
            pl.when(pl.col('pos').str.contains(pos)).then(pl.col(f"dropoff_{pos}_{pick}")).otherwise(pl.lit(0)).alias(
                f"dropoff_{pos}_{pick}")
            for pos in ['C', 'LW', 'RW', 'D', 'G']
            for pick in [1, 2, 3]]
        )
        .with_columns([
            pl.max_horizontal([cs.contains(f'_{pick}').and_(cs.contains('dropoff'))]).alias(f'dropoff_{pick}') for pick
            in
            [1, 2, 3]]
        )
        .with_columns(
            pl.col('vorp')
            .add(0.25*pl.col('dropoff_1'))
            .add(0.2*pl.col('dropoff_2'))
            .add(0.1*pl.col('dropoff_3'))
            .alias('scaled')
        )
    )
