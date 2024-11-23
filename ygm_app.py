import streamlit as st
import pandas as pd
import polars as pl
from yfh.functions import get_pick_numbers
from yfh.data import load_data, process_probs, process_dropoffs

# Initialize session state
if 'current_pick' not in st.session_state:
    st.session_state.current_pick = 1
if 'drafted_ids' not in st.session_state:
    st.session_state.drafted_ids = []
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""


# Load Data
@st.cache_data
def load_cached_data():
    return load_data('data/splunk_projections.xlsx', sheet_name='The List')


data = load_cached_data()

# Sidebar for draft setup
with st.sidebar:
    st.header("Draft Setup")
    NUM_TEAMS = st.number_input('# Teams', min_value=1, max_value=20, value=14)
    MY_PICK = st.number_input('Draft Slot', min_value=1, max_value=NUM_TEAMS, value=1)
    NUM_ROUNDS = st.number_input('# Rounds', min_value=10, max_value=30, value=16)

    all_picks = get_pick_numbers(NUM_TEAMS, MY_PICK, NUM_ROUNDS)
    next_picks = [pick for pick in all_picks if pick > st.session_state.current_pick][:3]

pick_1, pick_2, pick_3 = next_picks[:3] + [0] * (3 - len(next_picks))
sort_column = '$' if st.session_state.current_pick in all_picks else 'adp'
desc = st.session_state.current_pick in all_picks

d_need_factor = 0.788
g_need_factor = 0.621803

col1, col2 = st.columns(2)
with col1:
    num_dmen_drafted = st.radio("Number of Defensemen Drafted", [0, 1, 2, 3, 4], horizontal=True)
with col2:
    num_goalies_drafted = st.radio("Number of Goalies Drafted", [0, 1, 2], horizontal=True)

st.write(f"Current Pick: {st.session_state.current_pick}")

# Process Data
def process_data(_data, _pick_1, _pick_2, _pick_3, _drafted_ids):
    return (
        _data
        .filter(~pl.col('id').is_in(_drafted_ids))
        .pipe(process_probs, _pick_1, _pick_2, _pick_3)
        .pipe(process_dropoffs)
        .with_columns(
            pl.when(pl.col('pos') == 'D').then(pl.col('dropoff_1') * d_need_factor ** num_dmen_drafted).otherwise(
                pl.col('dropoff_1')).alias('dropoff_1'),
            pl.when(pl.col('pos') == 'D').then(pl.col('dropoff_2') * d_need_factor ** num_dmen_drafted).otherwise(
                pl.col('dropoff_2')).alias('dropoff_2'),
            pl.when(pl.col('pos') == 'D').then(pl.col('dropoff_3') * d_need_factor ** num_dmen_drafted).otherwise(
                pl.col('dropoff_3')).alias('dropoff_3'),
        )
        .with_columns(
            pl.when(pl.col('pos') == 'G').then(pl.col('dropoff_1') * g_need_factor ** num_goalies_drafted).otherwise(
                pl.col('dropoff_1')).alias('dropoff_1'),
            pl.when(pl.col('pos') == 'G').then(pl.col('dropoff_2') * g_need_factor ** num_goalies_drafted).otherwise(
                pl.col('dropoff_2')).alias('dropoff_2'),
            pl.when(pl.col('pos') == 'G').then(pl.col('dropoff_3') * g_need_factor ** num_goalies_drafted).otherwise(
                pl.col('dropoff_3')).alias('dropoff_3'),
        )
        .with_columns(
            (pl.col('vorp') + 0.3 * pl.col('dropoff_1') + 0.2 * pl.col('dropoff_2') + 0.1 * pl.col('dropoff_3')).alias(
                'scaled')
        )
        .select(
            pl.lit(False).alias('drafted'),
            pl.col('player'),
            pl.col('id'),
            pl.col('pos'),
            pl.col('adp'),
            pl.col('scaled').alias('$'),
            pl.col('vorp').alias('v'),
            pl.col('dropoff_1').alias('d1'),
            pl.col('dropoff_2').alias('d2'),
            pl.col('dropoff_3').alias('d3'),
            pl.col('prob_1').alias('p1'),
            pl.col('prob_2').alias('p2')
        )
        .sort(sort_column, descending=desc)
    )


# Use a unique key for processed data to force recomputation
processed = process_data(data, pick_1, pick_2, pick_3, tuple(st.session_state.drafted_ids))

# Convert to pandas and style
pandas_df = (
    processed
    .to_pandas()
)

# Search functionality
st.text_input("Search for a player:", key="search_query")
filtered_df = pandas_df[pandas_df['player'].str.contains(st.session_state.search_query, case=False)]


def highlight_max_row(row):
    # Check if the current row has the max value for both $ and v
    is_max_row = (row['$'] == filtered_df['$'].max()) and (row['v'] == filtered_df['v'].max())
    return ['background-color: lightblue' if is_max_row else ''] * len(row)


# Data editor with filtered results
edited_df = st.data_editor(
    filtered_df
    .style
    .format(precision=0, subset=['adp', 'v', 'd1', 'd2', 'd3', '$'])
    .format(precision=1, subset=['adp'])
    .format({'p1': "{:.0%}", 'p2': "{:.0%}"}, subset=['p1', 'p2'])
    .background_gradient(cmap='RdYlGn', vmin=processed['v'].min(), vmax=processed['v'].max(), subset=['v'])
    .background_gradient(cmap='RdYlGn', vmin=processed['d1'].min(), vmax=processed['d1'].max(), subset=['d1'])
    .background_gradient(cmap='RdYlGn', vmin=processed['d2'].min(), vmax=processed['d2'].max(), subset=['d2'])
    .background_gradient(cmap='RdYlGn', vmin=processed['d3'].min(), vmax=processed['d3'].max(), subset=['d3'])
    .background_gradient(cmap='RdYlGn', vmin=processed['$'].min(), vmax=processed['$'].max(), subset=['$'])
    .background_gradient(cmap='RdYlGn', vmin=0, vmax=processed['p1'].max(), subset=['p1'])
    .background_gradient(cmap='RdYlGn', vmin=0, vmax=processed['p2'].max(), subset=['p2'])
    .apply(highlight_max_row, axis=1)
    ,
    column_config={
        "drafted": st.column_config.CheckboxColumn(
            "D",
            help="Select drafted players",
            default=False,
        )
    },
    disabled=list(set(filtered_df.columns) - {'drafted'}),
    key=f"data_editor_{st.session_state.current_pick}",
    hide_index=True,
    use_container_width=True,
    height=650
)

# Create two columns for buttons
col1, col2 = st.columns(2)

# Handle draft selections
if col1.button("Confirm Selections"):
    newly_drafted = edited_df[edited_df['drafted'] == True]
    new_drafted_ids = newly_drafted['id'].tolist()
    st.session_state.drafted_ids.extend(new_drafted_ids)
    st.session_state.current_pick += len(new_drafted_ids)

    if st.session_state.current_pick > NUM_ROUNDS * NUM_TEAMS:
        st.write("Draft complete!")
    else:
        st.rerun()

# New button to skip to next pick without altering the dataframe
if col2.button("Skip to Next Pick"):
    st.session_state.current_pick += 1
    if st.session_state.current_pick > NUM_ROUNDS * NUM_TEAMS:
        st.write("Draft complete!")
    else:
        st.rerun()

# Display drafted players
st.write("Drafted Players:")
st.write(st.session_state.drafted_ids)

# Display number of available players
st.write(f"Number of available players: {len(processed)}")
