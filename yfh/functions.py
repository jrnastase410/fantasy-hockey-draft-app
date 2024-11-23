import polars as pl
from scipy import stats


def get_pick_numbers(num_teams_in_draft, my_pick_slot, num_rounds):
    """
    For a serpentine draft, return the pick numbers for the team that starts at `my_pick_slot`.

    Args:
        num_teams_in_draft (int): The total number of teams in the draft.
        my_pick_slot (int): Your pick position in the first round (1-indexed).
        num_rounds (int): The total number of rounds in the draft.

    Returns:
        list: A list of pick numbers for each round.
    """
    picks = []

    for round_num in range(1, num_rounds + 1):
        if round_num % 2 != 0:
            # Odd rounds: normal order
            pick_number = (round_num - 1) * num_teams_in_draft + my_pick_slot
        else:
            # Even rounds: reverse order
            pick_number = round_num * num_teams_in_draft - (my_pick_slot - 1)

        picks.append(pick_number)

    return picks


def availability_probability(row):
    adp, adp_sd, pick = row['adp'], row['adp_sd'], row['pick']
    if adp_sd == 0:  # Prevent division by zero for players with no variance
        return 1.0 if pick < adp else 0.0
    return 1 - stats.norm.cdf(pick, loc=adp, scale=adp_sd)


def get_next_round_expectation(df, position, pick_number):
    return (
        df
        .filter(pl.col('pos').str.contains(position))
        .with_columns(
            (1 - pl.col(f"prob_{pick_number}")).alias("prob_drafted")  # Probability the player is drafted
        )
        .with_columns(
            (pl.col("prob_drafted").cum_prod().shift(1, fill_value=1)).alias("prob_all_above_drafted")
        )
        .with_columns(
            (pl.col("prob_all_above_drafted") * pl.col(f"prob_{pick_number}")).alias("prob_best_available")
        )
        .with_columns(
            (pl.col('vorp') * pl.col('prob_best_available')).alias('vorp_weighted')
        )
        .select(pl.sum('vorp_weighted')).item()
    )
