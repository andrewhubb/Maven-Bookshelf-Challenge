import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Maven Bookshelf — Hardware Showdown",
    page_icon="📚",
    layout="wide",
)

# ── Machines ──────────────────────────────────────────────────────────────────

MACHINES = {
    "MacBook Air (M5 · 16GB)": {
        "file": "pipeline_log_20260513_051913-Macbook.csv",
        "color": "#635BFF",
        "color_fill": "rgba(99,91,255,0.13)",
        "chip": "Apple M5 · 16 GB · MPS",
        "short": "MacBook Air",
    },
    "Mac Mini (M4 · 32GB)": {
        "file": "pipeline_log_20260514_055106-Mac-Mini.csv",
        "color": "#00B4D8",
        "color_fill": "rgba(0,180,216,0.13)",
        "chip": "Apple M4 · 32 GB · MPS",
        "short": "Mac Mini",
    },
    "Asus TUF (RTX 3050 · 64GB)": {
        "file": "pipeline_log_20260515_054106_asus.csv",
        "color": "#FF6B35",
        "color_fill": "rgba(255,107,53,0.13)",
        "chip": "RTX 3050 Laptop · 4 GB VRAM · CUDA",
        "short": "Asus TUF",
    },
}

STAGE_LABELS = {
    "sentiment": "Sentiment (BERT)",
    "vader":     "VADER (rule-based)",
    "emotion":   "Emotion (BERT)",
    "db_write":  "DB Write",
}

DATASET_LABELS = {
    "works":   "Works  (~14k rows)",
    "reviews": "Reviews  (~115k rows)",
}

NORM_ORIGIN = pd.Timestamp("2026-01-01 05:00:00")

# ── Font sizes — single source of truth ──────────────────────────────────────
# Change these values to rescale everything at once.

FS_TICK   = 16   # axis tick labels
FS_TITLE  = 18   # axis titles
FS_BASE   = 16   # general chart text (hover labels, bar values, etc.)
FS_LEGEND = 16   # legend text
FS_ANNOT  = 15   # in-chart text annotations (finish lines, stage labels)

# CSS rem sizes for HTML sections
CSS_CARD_NAME   = "1.5rem"    # machine name in KPI card
CSS_CARD_CHIP   = "1.1rem"    # chip spec line
CSS_CARD_TIME   = "3.5rem"    # big time number
CSS_CARD_VS     = "1.15rem"   # "1.3× slower" line
CSS_CARD_SAVED  = "1.05rem"   # "+3h vs winner" line
CSS_HERO_LABEL  = "1.0rem"    # Pipeline / Dataset / Machines label
CSS_HERO_VALUE  = "1.15rem"   # Pipeline / Dataset / Machines value
CSS_HERO_BODY   = "1.15rem"   # hero paragraph
CSS_TAKEAWAY    = "1.05rem"   # takeaway body text
CSS_FOOTER      = "0.95rem"   # footer note

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    frames = []
    for name, meta in MACHINES.items():
        df = pd.read_csv(meta["file"])
        df["machine"]   = name
        df["color"]     = meta["color"]
        df["short"]     = meta["short"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


raw = load_data()

# Normalise: shift each machine so its pipeline start = 05:00
pipeline_starts = (
    raw[raw["event"] == "PIPELINE_START"]
    .groupby("machine")["timestamp"].min()
)

def normalise(df):
    starts = df["machine"].map(pipeline_starts)
    df["norm_time"] = NORM_ORIGIN + (df["timestamp"] - starts)
    return df

raw = normalise(raw)

chunks = raw[raw["event"] == "CHUNK_COMPLETE"].copy()
chunks["records_per_s"] = 1000 / chunks["duration_s"]
chunks["stage_label"]   = chunks["stage"].map(STAGE_LABELS)
chunks["dataset_label"] = chunks["dataset"].map(DATASET_LABELS)

stage_ends = raw[raw["event"] == "STAGE_END"].copy()

# Pipeline totals
totals = (
    stage_ends.groupby("machine")["duration_s"]
    .sum().reset_index()
    .rename(columns={"duration_s": "total_s"})
)
totals["total_h"] = totals["total_s"] / 3600
totals = totals.merge(
    pd.DataFrame([{"machine": k, **v} for k, v in MACHINES.items()]),
    on="machine",
).sort_values("total_h").reset_index(drop=True)

winner_h    = totals["total_h"].iloc[0]
slowest_h   = totals["total_h"].iloc[-1]
speedup     = slowest_h / winner_h
hours_saved = slowest_h - winner_h

# ── Shared chart helpers ──────────────────────────────────────────────────────

def time_axis(dtick=3_600_000):
    """Return x-axis dict for normalised HH:MM datetime axes."""
    return dict(
        tickformat="%H:%M",
        dtick=dtick,
        gridcolor="rgba(128,128,128,0.15)",
        tickfont=dict(size=FS_TICK),
        title_font=dict(size=FS_TITLE),
        minor=dict(dtick=dtick // 2, showgrid=True,
                   gridcolor="rgba(128,128,128,0.07)"),
    )

def base_layout(hovermode="x unified", **extra):
    """Common layout kwargs for every chart."""
    return dict(
        hovermode=hovermode,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20),
        font=dict(size=FS_BASE),
        legend=dict(
            font=dict(size=FS_LEGEND),
            title_font=dict(size=FS_LEGEND),
            title="Machine",
        ),
        **extra,
    )

def yaxis_style(title):
    return dict(
        title=title,
        title_font=dict(size=FS_TITLE),
        tickfont=dict(size=FS_TICK),
        gridcolor="rgba(128,128,128,0.15)",
    )

def xaxis_style(title, dtick=3_600_000):
    return dict(**time_axis(dtick), title=title)


# ════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div style='
        background: linear-gradient(135deg, rgba(255,107,53,0.12), rgba(99,91,255,0.08));
        border: 1.5px solid rgba(255,107,53,0.4);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 8px;
    '>
        <div style='font-size:1.0rem;color:#FF6B35;font-weight:700;
                    letter-spacing:.12em;text-transform:uppercase;margin-bottom:6px'>
            Maven Bookshelf Challenge · Hardware Showdown
        </div>
        <h1 style='margin:0 0 8px 0;font-size:2.0rem;line-height:1.2'>
            4 GB VRAM. Budget laptop GPU. Beats Apple Silicon.
        </h1>
        <p style='color:#aaa;max-width:820px;margin:0 0 20px 0;font-size:{CSS_HERO_BODY}'>
            Running the same NLP pipeline — BERT sentiment, BERT emotion, VADER —
            across <b>115,000+ book reviews</b> on three machines.
            The result? An entry-level RTX 3050 Laptop GPU finished
            <b style='color:#FF6B35'>{speedup:.1f}× faster</b> than the Mac Mini desktop,
            saving <b style='color:#FF6B35'>{hours_saved:.1f} hours</b>.
        </p>
        <div style='display:flex;gap:40px;flex-wrap:wrap'>
            <div>
                <span style='font-size:{CSS_HERO_LABEL};color:#888;display:block'>Pipeline</span>
                <span style='font-size:{CSS_HERO_VALUE};font-weight:600'>BERT · VADER · MySQL write</span>
            </div>
            <div>
                <span style='font-size:{CSS_HERO_LABEL};color:#888;display:block'>Dataset</span>
                <span style='font-size:{CSS_HERO_VALUE};font-weight:600'>115k+ book reviews</span>
            </div>
            <div>
                <span style='font-size:{CSS_HERO_LABEL};color:#888;display:block'>Machines</span>
                <span style='font-size:{CSS_HERO_VALUE};font-weight:600'>MacBook Air M5 · Mac Mini M4 · Asus TUF RTX 3050</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ════════════════════════════════════════════════════════════════════════════

st.subheader("Total Pipeline Time")

rank_medals = ["🥇", "🥈", "🥉"]
cols = st.columns(len(MACHINES))

for i, row in totals.iterrows():
    h = int(row["total_h"])
    m = int((row["total_h"] - h) * 60)
    ratio     = row["total_h"] / winner_h
    vs_winner = f"{ratio:.2f}× slower than winner" if i > 0 else "⚡ Fastest"
    vs_color  = "#888" if i > 0 else row["color"]
    saved     = row["total_h"] - winner_h
    saved_str = f"+{saved:.1f}h vs winner" if i > 0 else "&nbsp;"

    with cols[i]:
        st.markdown(
            f"""
            <div style='
                background:{row["color"]}15;
                border:1.5px solid {row["color"]};
                border-radius:14px;
                padding:28px 20px;
                text-align:center;
                height:280px;
                box-sizing:border-box;
                display:flex;
                flex-direction:column;
                align-items:center;
                justify-content:center;
                gap:6px;
            '>
                <div style='font-size:2.4rem'>{rank_medals[i]}</div>
                <div style='font-size:{CSS_CARD_NAME};font-weight:700'>{row["short"]}</div>
                <div style='font-size:{CSS_CARD_CHIP};color:#aaa'>{row["chip"]}</div>
                <div style='font-size:{CSS_CARD_TIME};font-weight:800;
                            color:{row["color"]};line-height:1.1;margin-top:4px'>{h}h {m}m</div>
                <div style='font-size:{CSS_CARD_VS};color:{vs_color};
                            font-weight:600;margin-top:4px'>{vs_winner}</div>
                <div style='font-size:{CSS_CARD_SAVED};color:#666'>{saved_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# RACE CHART — cumulative records processed over time
# ════════════════════════════════════════════════════════════════════════════

st.subheader("The Race: Cumulative Records Processed")
st.caption(
    "Each step = 1,000 records. Watch the Asus TUF pull away early "
    "and never look back. Vertical lines mark major stage transitions."
)

race = chunks.sort_values(["machine", "norm_time"]).copy()
race["cum_records"] = race.groupby("machine").cumcount().add(1).mul(1000)

stage_starts_raw = raw[raw["event"] == "STAGE_START"].copy()
key_stages = [
    ("reviews", "sentiment", "Reviews begin"),
    ("reviews", "emotion",   "Emotion stage"),
]

fig_race = go.Figure()

for name, meta in reversed(list(MACHINES.items())):
    md = race[race["machine"] == name]
    fig_race.add_trace(go.Scatter(
        x=md["norm_time"],
        y=md["cum_records"] / 1000,
        mode="lines",
        name=meta["short"],
        line=dict(color=meta["color"], width=3),
        fill="tozeroy",
        fillcolor=meta["color_fill"],
        hovertemplate=(
            f"<b>{meta['short']}</b><br>"
            "Time: %{x|%H:%M}<br>"
            "Records: %{y:.0f}k<extra></extra>"
        ),
    ))

y_top = race["cum_records"].max() / 1000 * 1.08

for ds, stage, label in key_stages:
    times = []
    for name in MACHINES:
        ev = stage_starts_raw[
            (stage_starts_raw["machine"] == name) &
            (stage_starts_raw["dataset"] == ds) &
            (stage_starts_raw["stage"] == stage)
        ]
        if not ev.empty:
            times.append(ev["norm_time"].iloc[0])
    if times:
        avg_t = pd.Timestamp(sum(t.value for t in times) // len(times))
        fig_race.add_trace(go.Scatter(
            x=[avg_t, avg_t], y=[0, y_top],
            mode="lines+text",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1),
            text=["", label],
            textposition="top center",
            textfont=dict(size=FS_ANNOT, color="#aaa"),
            showlegend=False, hoverinfo="skip",
        ))

for _, row in totals.iterrows():
    finish_t = NORM_ORIGIN + pd.Timedelta(seconds=float(row["total_s"]))
    fig_race.add_trace(go.Scatter(
        x=[finish_t, finish_t], y=[0, y_top],
        mode="lines+text",
        line=dict(color=row["color"], dash="dot", width=1.5),
        text=["", f"{row['short']} done"],
        textposition="top center",
        textfont=dict(size=FS_ANNOT, color=row["color"]),
        showlegend=False, hoverinfo="skip",
    ))

fig_race.update_layout(
    height=460,
    xaxis=xaxis_style("Time into run  (normalised to 05:00 start)"),
    yaxis=yaxis_style("Cumulative records processed (thousands)"),
    **base_layout(),
)
st.plotly_chart(fig_race, width="stretch")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# STAGE BREAKDOWN BAR CHART
# ════════════════════════════════════════════════════════════════════════════

st.subheader("Time by Stage")

tab_works, tab_reviews = st.tabs(["Works dataset  (~14k rows)", "Reviews dataset  (~115k rows)"])

for tab, ds_key in [(tab_works, "works"), (tab_reviews, "reviews")]:
    with tab:
        ds_data = stage_ends[
            (stage_ends["dataset"] == ds_key) &
            (stage_ends["stage"]   != "db_write")
        ].copy()
        ds_data["stage_label"] = ds_data["stage"].map(STAGE_LABELS)
        ds_data["duration_h"]  = ds_data["duration_s"] / 3600

        fig_bar = go.Figure()
        for name, meta in MACHINES.items():
            m_data = ds_data[ds_data["machine"] == name].sort_values("stage")
            fig_bar.add_trace(go.Bar(
                name=meta["short"],
                x=m_data["stage_label"],
                y=m_data["duration_h"],
                marker_color=meta["color"],
                text=[f"{v:.2f}h" if v >= 0.1 else f"{v*60:.1f}m"
                      for v in m_data["duration_h"]],
                textposition="outside",
                textfont=dict(size=FS_BASE),
            ))

        fig_bar.update_layout(
            barmode="group",
            height=460,
            xaxis=dict(
                title="Pipeline Stage",
                title_font=dict(size=FS_TITLE),
                tickfont=dict(size=FS_TICK),
            ),
            yaxis=yaxis_style("Hours"),
            **base_layout(hovermode="closest"),
        )
        st.plotly_chart(fig_bar, width="stretch")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# CHUNK-BY-CHUNK SPEED
# ════════════════════════════════════════════════════════════════════════════

st.subheader("Chunk-by-Chunk Speed")
st.caption(
    "Each chunk = 1,000 records. Higher records/sec = faster inference. "
    "Dashed vertical lines mark when each machine finishes that stage."
)

col_f1, col_f2 = st.columns(2)
with col_f1:
    selected_dataset = st.selectbox(
        "Dataset",
        options=list(DATASET_LABELS.keys()),
        format_func=lambda x: DATASET_LABELS[x],
    )
with col_f2:
    selected_stage = st.selectbox(
        "Stage",
        options=["sentiment", "emotion"],
        format_func=lambda x: STAGE_LABELS[x],
    )

plot_data = chunks[
    (chunks["dataset"] == selected_dataset) &
    (chunks["stage"]   == selected_stage)
].copy()
plot_data = plot_data.sort_values(["machine", "chunk_id"])
plot_data["chunk_seq"] = plot_data.groupby("machine").cumcount()

tab_ts, tab_bar2 = st.tabs(["Time Series", "By Chunk"])

with tab_ts:
    fig_ts = go.Figure()

    for name, meta in MACHINES.items():
        md = plot_data[plot_data["machine"] == name].sort_values("norm_time")
        fig_ts.add_trace(go.Scatter(
            x=md["norm_time"],
            y=md["records_per_s"],
            mode="lines+markers",
            name=meta["short"],
            line=dict(color=meta["color"], width=2.5),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{meta['short']}</b><br>"
                "Time: %{x|%H:%M}<br>"
                "Records/s: %{y:.1f}<br>"
                "Chunk: %{customdata}<extra></extra>"
            ),
            customdata=md["chunk_id"],
        ))

    stage_finish = stage_ends[
        (stage_ends["dataset"] == selected_dataset) &
        (stage_ends["stage"]   == selected_stage)
    ]
    y_ts_top = plot_data["records_per_s"].max() * 1.12
    for _, sf in stage_finish.iterrows():
        meta = MACHINES[sf["machine"]]
        fig_ts.add_trace(go.Scatter(
            x=[sf["norm_time"], sf["norm_time"]],
            y=[0, y_ts_top],
            mode="lines+text",
            line=dict(color=meta["color"], dash="dash", width=1.5),
            text=["", f"{meta['short']} done"],
            textposition="top center",
            textfont=dict(size=FS_ANNOT, color=meta["color"]),
            showlegend=False, hoverinfo="skip",
        ))

    # Adaptive x range and tick density
    data_start = plot_data["norm_time"].min()
    data_end   = plot_data["norm_time"].max()
    finish_end = stage_finish["norm_time"].max() if not stage_finish.empty else data_end
    range_secs = (max(data_end, finish_end) - data_start).total_seconds()

    if range_secs < 1800:
        pad, ts_dtick = pd.Timedelta(minutes=2), 5 * 60 * 1000
    elif range_secs < 7200:
        pad, ts_dtick = pd.Timedelta(minutes=10), 30 * 60 * 1000
    else:
        pad, ts_dtick = pd.Timedelta(minutes=15), 3_600_000

    x_min = data_start - pad
    x_max = max(data_end, finish_end) + pad

    fig_ts.update_layout(
        height=500,
        xaxis=dict(
            **xaxis_style(
                "Time into run  (all machines normalised to 05:00 start)",
                dtick=ts_dtick,
            ),
            range=[x_min.isoformat(), x_max.isoformat()],
            autorange=False,
        ),
        yaxis=yaxis_style("Records / second"),
        **base_layout(),
    )
    st.plotly_chart(fig_ts, width="stretch")
    st.caption(
        "Timestamps normalised to a common 05:00 origin. "
        "Each machine actually ran on a different day."
    )

with tab_bar2:
    fig_bar2 = go.Figure()
    for name, meta in MACHINES.items():
        md = plot_data[plot_data["machine"] == name]
        fig_bar2.add_trace(go.Bar(
            name=meta["short"],
            x=md["chunk_seq"],
            y=md["records_per_s"],
            marker_color=meta["color"],
            opacity=0.85,
            hovertemplate=(
                f"<b>{meta['short']}</b><br>"
                "Chunk #%{x}<br>"
                "Records/s: %{y:.1f}<extra></extra>"
            ),
        ))

    fig_bar2.update_layout(
        barmode="group",
        height=460,
        xaxis=dict(
            title="Chunk #",
            title_font=dict(size=FS_TITLE),
            tickfont=dict(size=FS_TICK),
        ),
        yaxis=yaxis_style("Records / second"),
        **base_layout(hovermode="closest"),
    )
    st.plotly_chart(fig_bar2, width="stretch")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# THROUGHPUT CONSISTENCY — rolling average
# ════════════════════════════════════════════════════════════════════════════

st.subheader("Throughput Consistency")
st.caption("10-chunk rolling mean. Reveals sustained performance vs burst behaviour.")

fig_smooth = go.Figure()
for name, meta in MACHINES.items():
    md = plot_data[plot_data["machine"] == name].sort_values("norm_time").copy()
    md["rolling"] = md["records_per_s"].rolling(window=10, min_periods=1).mean()
    fig_smooth.add_trace(go.Scatter(
        x=md["norm_time"],
        y=md["rolling"],
        mode="lines",
        name=meta["short"],
        line=dict(color=meta["color"], width=3),
        fill="tozeroy",
        fillcolor=meta["color_fill"],
        hovertemplate=(
            f"<b>{meta['short']}</b><br>"
            "Time: %{x|%H:%M}<br>"
            "Avg records/s: %{y:.1f}<extra></extra>"
        ),
    ))

fig_smooth.update_layout(
    height=400,
    xaxis=xaxis_style(
        "Time into run  (normalised to 05:00 start)",
        dtick=ts_dtick,
    ),
    yaxis=yaxis_style("Records / second (10-chunk rolling avg)"),
    **base_layout(),
)
st.plotly_chart(fig_smooth, width="stretch")

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

st.subheader("Head-to-Head Summary")

rows = []
for name, meta in MACHINES.items():
    for ds in ["works", "reviews"]:
        for stage in ["sentiment", "emotion"]:
            d = chunks[
                (chunks["machine"] == name) &
                (chunks["dataset"] == ds) &
                (chunks["stage"]   == stage)
            ]
            if d.empty:
                continue
            rows.append({
                "Machine":        meta["short"],
                "Dataset":        DATASET_LABELS[ds],
                "Stage":          STAGE_LABELS[stage],
                "Avg records/s":  f"{d['records_per_s'].mean():.1f}",
                "Peak records/s": f"{d['records_per_s'].max():.1f}",
                "Chunks":         len(d),
            })

st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════
# KEY TAKEAWAYS
# ════════════════════════════════════════════════════════════════════════════

st.subheader("Key Takeaways")

asus_color = MACHINES["Asus TUF (RTX 3050 · 64GB)"]["color"]
air_color  = MACHINES["MacBook Air (M5 · 16GB)"]["color"]
mini_color = MACHINES["Mac Mini (M4 · 32GB)"]["color"]

col1, col2, col3 = st.columns(3)

_ta = f"font-size:{CSS_TAKEAWAY};line-height:1.7"   # takeaway body style

with col1:
    st.markdown(
        f"""
        <div style='border-left:4px solid {asus_color};padding-left:16px'>
        <p style='font-size:1.15rem;font-weight:700;margin:0 0 10px 0'>
            🟠 Budget GPU wins — by a mile
        </p>
        <p style='{_ta};margin:0'>
            The RTX 3050 Laptop has just <b>4 GB of VRAM</b>, yet CUDA's mature
            PyTorch backend drives it to finish in <b>~10h</b> — {speedup:.1f}× faster
            than the Mac Mini. An entry-level gaming GPU outrunning dedicated Apple Silicon
            is the headline result of this experiment.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style='border-left:4px solid {air_color};padding-left:16px'>
        <p style='font-size:1.15rem;font-weight:700;margin:0 0 10px 0'>
            🟣 MacBook Air M5 punches above its weight
        </p>
        <p style='{_ta};margin:0'>
            With only <b>16 GB unified memory</b> — half the Mac Mini's RAM — the M5 Air
            finishes in <b>~13h</b>, comfortably ahead of the desktop. The newer M5 chip
            and a well-optimised MPS stack deliver consistent throughput on a thin-and-light
            with no active cooling.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div style='border-left:4px solid {mini_color};padding-left:16px'>
        <p style='font-size:1.15rem;font-weight:700;margin:0 0 10px 0'>
            🔵 Mac Mini M4 disappoints on sustained inference
        </p>
        <p style='{_ta};margin:0'>
            More RAM (32 GB), a desktop chassis, a newer-generation chip — and yet
            the slowest result at <b>~18h</b>. The M4's MPS backend appears to throttle
            harder on long BERT workloads than the M5. More spec on paper ≠ faster
            inference in practice.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div style='color:#666;font-size:{CSS_FOOTER};margin-top:32px;line-height:2'>
    Pipeline: BERT sentiment &nbsp;·&nbsp; BERT emotion &nbsp;·&nbsp; VADER &nbsp;·&nbsp; MySQL write
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Dataset: Open Library / Goodreads book reviews
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Machines: MacBook Air M5 16 GB &nbsp;·&nbsp; Mac Mini M4 32 GB &nbsp;·&nbsp; Asus TUF RTX 3050 Laptop 4 GB VRAM 64 GB RAM
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Built for the Maven Bookshelf Challenge
    </div>
    """,
    unsafe_allow_html=True,
)
