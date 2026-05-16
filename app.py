import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Maven Bookshelf — Hardware Showdown",
    page_icon="📚",
    layout="wide",
)

# ── Data ─────────────────────────────────────────────────────────────────────

MACHINES = {
    "MacBook Air (M5 · 16GB)": {
        "file": "pipeline_log_20260513_051913-Macbook.csv",
        "color": "#635BFF",             # purple
        "color_fill": "rgba(99,91,255,0.13)",
        "chip": "Apple M5 · 16 GB · MPS",
        "short": "MacBook Air",
    },
    "Mac Mini (M4 · 32GB)": {
        "file": "pipeline_log_20260514_055106-Mac-Mini.csv",
        "color": "#00B4D8",             # cyan
        "color_fill": "rgba(0,180,216,0.13)",
        "chip": "Apple M4 · 32 GB · MPS",
        "short": "Mac Mini",
    },
    "Asus TUF (RTX 3050 · 64GB)": {
        "file": "pipeline_log_20260515_054106_asus.csv",
        "color": "#FF6B35",             # orange
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


@st.cache_data
def load_data():
    frames = []
    for name, meta in MACHINES.items():
        df = pd.read_csv(meta["file"])
        df["machine"] = name
        df["color"]   = meta["color"]
        df["short"]   = meta["short"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


raw = load_data()
chunks = raw[raw["event"] == "CHUNK_COMPLETE"].copy()
chunks["records_per_s"] = 1000 / chunks["duration_s"]
chunks["stage_label"]   = chunks["stage"].map(STAGE_LABELS)
chunks["dataset_label"] = chunks["dataset"].map(DATASET_LABELS)

stage_times = raw[raw["event"] == "STAGE_END"].copy()
stage_times["stage_label"]   = stage_times["stage"].map(STAGE_LABELS)
stage_times["dataset_label"] = stage_times["dataset"].map(DATASET_LABELS)

# pipeline totals
totals = (
    stage_times.groupby("machine")["duration_s"]
    .sum()
    .reset_index()
    .rename(columns={"duration_s": "total_s"})
)
totals["total_h"] = totals["total_s"] / 3600
totals = totals.merge(
    pd.DataFrame([{"machine": k, **v} for k, v in MACHINES.items()]),
    on="machine",
)
totals = totals.sort_values("total_h")


# ── Helpers ───────────────────────────────────────────────────────────────────

def machine_color(name):
    return MACHINES[name]["color"]


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <h1 style='margin-bottom:0'>📚 Maven Bookshelf Challenge</h1>
    <h3 style='margin-top:4px;color:#888'>Hardware Showdown: Apple Silicon vs NVIDIA CUDA</h3>
    <p style='color:#aaa;max-width:780px'>
    Running the same NLP pipeline (BERT sentiment + emotion + VADER) across <b>115,000+ book reviews</b>
    on three very different machines — a MacBook Air M5, a Mac Mini M4, and an Asus TUF with an
    entry-level RTX 3050 Laptop GPU (4 GB VRAM). Who finishes first?
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── KPI cards ─────────────────────────────────────────────────────────────────

st.subheader("Total Pipeline Time")

cols = st.columns(len(MACHINES))
rank_medals = ["🥇", "🥈", "🥉"]

for i, (_, row) in enumerate(totals.iterrows()):
    medal = rank_medals[i]
    h = int(row["total_h"])
    m = int((row["total_h"] - h) * 60)
    with cols[i]:
        st.markdown(
            f"""
            <div style='
                background:{row["color"]}18;
                border:1.5px solid {row["color"]};
                border-radius:12px;
                padding:20px 24px;
                text-align:center;
            '>
                <div style='font-size:2rem'>{medal}</div>
                <div style='font-size:1.15rem;font-weight:700'>{row["short"]}</div>
                <div style='font-size:0.8rem;color:#aaa;margin-bottom:8px'>{row["chip"]}</div>
                <div style='font-size:2.4rem;font-weight:800;color:{row["color"]}'>{h}h {m}m</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Stage breakdown bar chart ─────────────────────────────────────────────────

st.subheader("Time by Stage")

tab_works, tab_reviews = st.tabs(["Works dataset", "Reviews dataset"])

for tab, ds_key in [(tab_works, "works"), (tab_reviews, "reviews")]:
    with tab:
        ds_data = stage_times[stage_times["dataset"] == ds_key].copy()
        ds_data["duration_h"] = ds_data["duration_s"] / 3600

        fig = go.Figure()
        for name, meta in MACHINES.items():
            m_data = ds_data[ds_data["machine"] == name].sort_values("stage")
            fig.add_trace(go.Bar(
                name=meta["short"],
                x=m_data["stage_label"],
                y=m_data["duration_h"],
                marker_color=meta["color"],
                text=[f"{v:.2f}h" if v >= 0.1 else f"{v*60:.1f}m"
                      for v in m_data["duration_h"]],
                textposition="outside",
            ))

        fig.update_layout(
            barmode="group",
            height=420,
            yaxis_title="Hours",
            xaxis_title="Pipeline Stage",
            legend_title="Machine",
            margin=dict(t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Chunk speed section ───────────────────────────────────────────────────────

st.subheader("Chunk-by-Chunk Speed")
st.caption(
    "Each chunk = 1,000 records. "
    "Higher records/sec means faster inference. "
    "The time-series view reveals warm-up, throttling, and consistency."
)

col_filter1, col_filter2 = st.columns([1, 1])
with col_filter1:
    selected_dataset = st.selectbox(
        "Dataset",
        options=list(DATASET_LABELS.keys()),
        format_func=lambda x: DATASET_LABELS[x],
    )
with col_filter2:
    selected_stage = st.selectbox(
        "Stage",
        options=["sentiment", "emotion"],
        format_func=lambda x: STAGE_LABELS[x],
    )

plot_data = chunks[
    (chunks["dataset"] == selected_dataset) &
    (chunks["stage"] == selected_stage)
].copy()

# add sequential chunk index per machine (so all machines start at 0)
plot_data = plot_data.sort_values(["machine", "chunk_id"])
plot_data["chunk_seq"] = plot_data.groupby("machine").cumcount()

tab_ts, tab_bar = st.tabs(["Time Series", "By Chunk"])

with tab_ts:
    fig_ts = go.Figure()
    for name, meta in MACHINES.items():
        md = plot_data[plot_data["machine"] == name]
        fig_ts.add_trace(go.Scatter(
            x=md["timestamp"],
            y=md["records_per_s"],
            mode="lines+markers",
            name=meta["short"],
            line=dict(color=meta["color"], width=2),
            marker=dict(size=4),
            hovertemplate=(
                f"<b>{meta['short']}</b><br>"
                "Time: %{x|%H:%M:%S}<br>"
                "Records/s: %{y:.1f}<br>"
                "Chunk: %{customdata}<extra></extra>"
            ),
            customdata=md["chunk_id"],
        ))

    fig_ts.update_layout(
        height=420,
        xaxis_title="Wall-clock Time",
        yaxis_title="Records / second",
        legend_title="Machine",
        hovermode="x unified",
        margin=dict(t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_ts.update_xaxes(showgrid=False)
    fig_ts.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig_ts, use_container_width=True)

with tab_bar:
    fig_bar = go.Figure()
    for name, meta in MACHINES.items():
        md = plot_data[plot_data["machine"] == name]
        fig_bar.add_trace(go.Bar(
            name=meta["short"],
            x=md["chunk_seq"],
            y=md["records_per_s"],
            marker_color=meta["color"],
            opacity=0.85,
        ))

    fig_bar.update_layout(
        barmode="group",
        height=420,
        xaxis_title="Chunk #",
        yaxis_title="Records / second",
        legend_title="Machine",
        margin=dict(t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_bar.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Rolling average smoothed view ────────────────────────────────────────────

st.subheader("Throughput Consistency (Rolling Average)")
st.caption("10-chunk rolling mean reveals sustained performance vs bursting behaviour.")

fig_smooth = go.Figure()
for name, meta in MACHINES.items():
    md = plot_data[plot_data["machine"] == name].sort_values("chunk_seq").copy()
    md["rolling"] = md["records_per_s"].rolling(window=10, min_periods=1).mean()

    fig_smooth.add_trace(go.Scatter(
        x=md["chunk_seq"],
        y=md["rolling"],
        mode="lines",
        name=meta["short"],
        line=dict(color=meta["color"], width=3),
        fill="tozeroy",
        fillcolor=meta["color_fill"],
    ))

fig_smooth.update_layout(
    height=350,
    xaxis_title="Chunk #",
    yaxis_title="Records / second (10-chunk avg)",
    legend_title="Machine",
    margin=dict(t=20, b=20),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig_smooth.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
st.plotly_chart(fig_smooth, use_container_width=True)

st.divider()

# ── Head-to-head summary table ────────────────────────────────────────────────

st.subheader("Head-to-Head Summary")

rows = []
for name, meta in MACHINES.items():
    for ds in ["works", "reviews"]:
        for stage in ["sentiment", "emotion"]:
            d = chunks[(chunks["machine"] == name) &
                       (chunks["dataset"] == ds) &
                       (chunks["stage"] == stage)]
            if d.empty:
                continue
            rows.append({
                "Machine": meta["short"],
                "Dataset": DATASET_LABELS[ds],
                "Stage": STAGE_LABELS[stage],
                "Avg records/s": f"{d['records_per_s'].mean():.1f}",
                "Peak records/s": f"{d['records_per_s'].max():.1f}",
                "Chunks": len(d),
            })

summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()

# ── Key takeaways ─────────────────────────────────────────────────────────────

st.subheader("Key Takeaways")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        **🟠 Asus TUF wins — with an entry-level GPU**
        The RTX 3050 Laptop GPU has just **4 GB of VRAM**, yet CUDA acceleration
        drives sentiment inference so fast the whole pipeline finishes in **~10h** —
        45% faster than the Mac Mini desktop. A budget laptop GPU beating dedicated
        Apple Silicon chips is the headline result.
        """
    )

with col2:
    st.markdown(
        """
        **🟣 MacBook Air M5 holds its own**
        With only **16 GB of unified memory** — half the Mac Mini's RAM —
        the M5 Air finishes in **~13h**, comfortably ahead of the Mac Mini.
        MPS on M5 delivers consistent per-chunk throughput despite the tighter
        memory budget. Great value for a thin-and-light.
        """
    )

with col3:
    st.markdown(
        """
        **🔵 Mac Mini M4 disappoints despite the spec advantage**
        More RAM (32 GB), a newer chip, a desktop form factor — and yet
        the slowest result at **~18h**. The M4's MPS backend appears to
        throttle differently on sustained BERT workloads. More headroom
        on paper doesn't always translate to faster inference in practice.
        """
    )

st.markdown(
    """
    <div style='color:#888;font-size:0.8rem;margin-top:24px'>
    Pipeline: BERT sentiment · BERT emotion · VADER · MySQL write ·
    Dataset: Open Library / Goodreads reviews ·
    Machines: MacBook Air M5 16 GB · Mac Mini M4 32 GB · Asus TUF RTX 3050 Laptop 4 GB VRAM 64 GB RAM ·
    Built for the <a href='#' style='color:#888'>Maven Bookshelf Challenge</a>
    </div>
    """,
    unsafe_allow_html=True,
)
