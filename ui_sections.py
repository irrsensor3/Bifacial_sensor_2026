import streamlit as st
import pandas as pd
from docx import Document
from io import BytesIO
from datetime import datetime
import os
import hashlib
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
from supabase import create_client

# Create a Supabase client here so other modules can import it from this package
supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

# =========================
# AUTHENTICATION
# =========================

SALT = "pv_secure_salt_2026"

def hash_password(password: str) -> str:
    return hashlib.sha256((password + SALT).encode()).hexdigest()


def check_password(password: str) -> bool:
    # Keep the simple hard-coded admin password check for now
    return hash_password(password) == hash_password("admin123")


def login():
    """Render the login UI and set session state on success."""
    st.title("🔐 Login")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Admin Login")
        password = st.text_input("Enter Admin Password", type="password", key="admin_pass")

        if st.button("Login as Admin"):
            if check_password(password):
                st.session_state.auth = True
                st.session_state.user_role = "admin"
                st.success("Admin access granted")
                st.rerun()
            else:
                st.error("Wrong password")

    with col2:
        st.subheader("Guest Access")
        if st.button("Login as Guest"):
            st.session_state.auth = True
            st.session_state.user_role = "guest"
            st.success("Guest access granted")
            st.rerun()


def require_login(role: str | None = None):
    """Page guard for the multipage app. Call this at the top of every
    page in app_pages/. app.py already gates access to st.navigation
    behind login (and hides the admin page from non-admins), so this
    is mainly a safety net for direct reruns. If `role` is given (e.g.
    "admin"), also requires that exact role and stops with an info
    message otherwise."""
    if not st.session_state.get("auth"):
        st.warning("Please log in first.")
        st.stop()

    if role is not None and st.session_state.get("user_role") != role:
        st.info(f"ℹ️ This page is only available in {role} mode.")
        st.stop()


def inject_theme():
    """Injects the 'Solar Admin' visual theme — light background,
    white rounded cards with soft shadows, blue/green/amber accent
    colors, clean sans-serif type. Called once from app.py; since
    st.navigation reruns app.py's whole script on every page switch,
    one call there covers every page, no per-page injection needed."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg: #F1F4F9;
            --card: #FFFFFF;
            --ink: #1F2937;
            --ink-muted: #6B7280;
            --blue: #3B82F6;
            --green: #22C55E;
            --amber: #F59E0B;
            --purple: #8B5CF6;
            --border: #E5E9F0;
        }

        .stApp {
            background-color: var(--bg);
            color: var(--ink);
        }

        section[data-testid="stSidebar"] {
            background-color: var(--card);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] * {
            font-family: 'Inter', sans-serif;
            color: var(--ink) !important;
        }
        section[data-testid="stSidebar"] a[aria-current="page"] {
            background-color: #EFF6FF;
            border-left: 3px solid var(--blue);
            border-radius: 6px;
        }

        h1, h2, h3 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            color: var(--ink) !important;
        }

        body, p, div, span, label {
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stMetricValue"] {
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
            color: var(--ink) !important;
        }
        [data-testid="stMetricLabel"] {
            font-family: 'Inter', sans-serif !important;
            color: var(--ink-muted) !important;
            font-size: 0.8rem !important;
        }

        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
        }

        .stButton > button {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.85rem;
            background-color: var(--blue);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 0.45rem 1rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
        }
        .stButton > button:hover {
            background-color: #2563EB;
            color: #FFFFFF;
        }
        .stButton > button:focus:not(:active) {
            outline: 2px solid var(--blue);
            outline-offset: 2px;
        }

        hr, div[data-testid="stDivider"] {
            border-top: 1px solid var(--border) !important;
        }

        .stTextInput input, .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] {
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--ink) !important;
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stCaptionContainer"] {
            font-family: 'Inter', sans-serif !important;
            color: var(--ink-muted) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_stamp(label: str):
    """Renders a small colored pill with the page name at the top of
    a page, in the Solar Admin theme's blue accent."""
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            font-family:'Inter', sans-serif;
            font-size:0.72rem;
            font-weight:600;
            letter-spacing:0.03em;
            text-transform:uppercase;
            color:#3B82F6;
            background-color:#EFF6FF;
            border-radius:20px;
            padding:4px 14px;
            margin-bottom:0.6rem;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_gauge(value, max_value, title, unit="", color="#3B82F6"):
    """Builds a Plotly speedometer-style gauge (mode='gauge+number')
    for a single live metric — e.g. current irradiance or panel power.
    Bands the gauge track into low/moderate/high thirds like a
    typical dashboard gauge, colored in `color`."""
    import plotly.graph_objects as go

    third = max_value / 3
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": f" {unit}", "font": {"size": 28}},
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, third], "color": "#FEE2E2"},
                    {"range": [third, third * 2], "color": "#FEF3C7"},
                    {"range": [third * 2, max_value], "color": "#DCFCE7"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#1F2937"},
    )
    return fig


def plot_line_chart(df, x_col, y_cols, x_range=None, y_range=None, y_title=""):
    """Multi-line Plotly chart (replaces st.line_chart for the Live
    Monitoring charts) so the axes can be forced to explicit
    min/max values instead of only auto-scaling. x_range/y_range are
    optional (min, max) tuples — pass None to leave that axis on
    autoscale."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(go.Scatter(x=df[x_col], y=df[col], mode="lines", name=str(col)))

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#1F2937"},
        legend={"orientation": "h", "y": -0.2},
        yaxis_title=y_title,
    )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


# =========================
# PLOTTING HELPERS
# =========================

def fig_to_image_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf


def plot_weather_signals(time, temperatures, irradiances, title="Weather Data", temp_ylim=None, irr_ylim=None):
    # A blank/missing cell in the Time column reads as NaN (a float),
    # which crashes matplotlib's categorical x-axis if it's mixed in
    # with real time strings. pandas' Series.astype(str) doesn't
    # reliably convert every value (some floats can slip through), so
    # convert element-by-element with plain Python str() instead.
    time = [str(t) for t in time]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    for label, temp_values in temperatures.items():
        ax1.plot(time, temp_values, label=str(label))

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature (°C)")
    if temp_ylim is not None:
        ax1.set_ylim(temp_ylim)

    ax2 = ax1.twinx()

    for label, irr_values in irradiances.items():
        ax2.plot(time, irr_values, linestyle="--", label=str(label))

    ax2.set_ylabel("Irradiance (W/m²)")
    if irr_ylim is not None:
        ax2.set_ylim(irr_ylim)

    plt.title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    return fig


def plot_irradiance_frequency(df, columns, bin_width=50, max_irr=1200):
    """Returns {column: matplotlib figure}, one histogram per column,
    showing how often that sensor's irradiance readings fell into each
    bin from 0 up to max_irr W/m². Non-numeric/out-of-range values are
    dropped before binning rather than raising."""
    bins = list(range(0, max_irr + bin_width, bin_width))
    figs = {}
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        values = values[(values >= 0) & (values <= max_irr)]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(values, bins=bins, color="#f5a623", edgecolor="black")
        ax.set_xlabel("Irradiance (W/m²)")
        ax.set_ylabel("Frequency")
        ax.set_title(str(col))
        ax.set_xlim(0, max_irr)
        fig.tight_layout()
        figs[col] = fig
    return figs


# =========================
# REPORT DATA BUILDERS & RENDERERS
# =========================

def build_report_data(df, report_title, observation, fig, df_dcm=None):
    """Build uniform report data structure. df_dcm is optional DC
    meter data (Datetime/Device_ID/... columns already standardized
    to created_at/device_id/... by drive_fetch) — when provided, adds
    a per-device DC meter summary table alongside the sensor summary."""
    start_time = f"{df['Date'].iloc[0]} {df['Time'].iloc[0]}" if "Date" in df.columns and "Time" in df.columns else "N/A"
    end_time = f"{df['Date'].iloc[-1]} {df['Time'].iloc[-1]}" if "Date" in df.columns and "Time" in df.columns else "N/A"

    metadata = [
        ("Generated Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Total Records", str(df.shape[0])),
        ("Total Columns", str(df.shape[1])),
        ("Start Time", start_time),
        ("End Time", end_time),
        ("Observation Notes", observation if observation else "")
    ]

    columns_list = ", ".join(df.columns)

    numeric_df = df.select_dtypes(include="number")
    numeric_summary = None

    if not numeric_df.empty:
        numeric_summary = []
        for col in numeric_df.columns:
            numeric_summary.append({
                "Column": col,
                "Mean": f"{numeric_df[col].mean():.2f}",
                "Min": f"{numeric_df[col].min():.2f}",
                "Max": f"{numeric_df[col].max():.2f}"
            })

    dcm_summary = None
    if df_dcm is not None and not df_dcm.empty and "device_id" in df_dcm.columns:
        metric_defs = [
            ("voltage_v", "Voltage (V)"),
            ("current_a", "Current (A)"),
            ("active_power_kw", "Power (kW)"),
            ("forward_energy_kwh", "Energy (kWh)"),
        ]
        dcm_summary = []
        for device_id, group in df_dcm.groupby("device_id"):
            device_label = f"Meter {int(device_id)}" if pd.notna(device_id) else "Unknown meter"
            for col, label in metric_defs:
                if col not in group.columns:
                    continue
                vals = pd.to_numeric(group[col], errors="coerce").dropna()
                if vals.empty:
                    continue
                dcm_summary.append({
                    "Column": f"{device_label} — {label}",
                    "Mean": f"{vals.mean():.2f}",
                    "Min": f"{vals.min():.2f}",
                    "Max": f"{vals.max():.2f}",
                })
        if not dcm_summary:
            dcm_summary = None

    return {
        "title": report_title,
        "metadata": metadata,
        "columns": columns_list,
        "numeric_summary": numeric_summary,
        "dcm_summary": dcm_summary,
        "figure": fig
    }


def preview_report_content(df, report_title, observation, fig, df_dcm=None):
    """Display a preview of what will be in the report"""
    report_data = build_report_data(df, report_title, observation, fig, df_dcm=df_dcm)

    # Report Title
    st.markdown(f"# {report_data['title']}")

    # Metadata Table
    metadata_df = pd.DataFrame(report_data['metadata'], columns=["Field", "Value"]) if report_data['metadata'] else pd.DataFrame()
    st.dataframe(metadata_df, use_container_width=True, hide_index=True)

    # Column Overview
    st.markdown("## Column Overview")
    st.write(report_data['columns'])

    # Numeric Summary
    st.markdown("## Numeric Summary")
    if report_data['numeric_summary']:
        summary_df = pd.DataFrame(report_data['numeric_summary'])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("No numeric columns found")

    # DC Meter Summary
    if report_data['dcm_summary']:
        st.markdown("## DC Meter Summary")
        dcm_df = pd.DataFrame(report_data['dcm_summary'])
        st.dataframe(dcm_df, use_container_width=True, hide_index=True)

    # Weather Graph
    st.markdown("## Weather Graph")
    st.pyplot(report_data['figure'])


def generate_word_report(df, report_title, observation, fig, df_dcm=None):
    report_data = build_report_data(df, report_title, observation, fig, df_dcm=df_dcm)

    doc = Document()
    doc.add_heading(report_data['title'], level=1)

    # Metadata Table
    table = doc.add_table(rows=len(report_data['metadata']), cols=2)
    table.style = "Table Grid"

    for i, (key, value) in enumerate(report_data['metadata']):
        table.cell(i, 0).text = key
        table.cell(i, 1).text = value

    # Column Overview
    doc.add_heading("Column Overview", level=2)
    doc.add_paragraph(report_data['columns'])

    # Numeric Summary
    if report_data['numeric_summary']:
        doc.add_heading("Numeric Summary", level=2)
        summary_table = doc.add_table(rows=len(report_data['numeric_summary']) + 1, cols=4)
        summary_table.style = "Table Grid"

        headers = ["Column", "Mean", "Min", "Max"]
        for col_idx, header in enumerate(headers):
            summary_table.cell(0, col_idx).text = header

        for row_idx, row_data in enumerate(report_data['numeric_summary'], start=1):
            summary_table.cell(row_idx, 0).text = row_data["Column"]
            summary_table.cell(row_idx, 1).text = row_data["Mean"]
            summary_table.cell(row_idx, 2).text = row_data["Min"]
            summary_table.cell(row_idx, 3).text = row_data["Max"]

    # DC Meter Summary
    if report_data['dcm_summary']:
        doc.add_heading("DC Meter Summary", level=2)
        dcm_table = doc.add_table(rows=len(report_data['dcm_summary']) + 1, cols=4)
        dcm_table.style = "Table Grid"

        headers = ["Column", "Mean", "Min", "Max"]
        for col_idx, header in enumerate(headers):
            dcm_table.cell(0, col_idx).text = header

        for row_idx, row_data in enumerate(report_data['dcm_summary'], start=1):
            dcm_table.cell(row_idx, 0).text = row_data["Column"]
            dcm_table.cell(row_idx, 1).text = row_data["Mean"]
            dcm_table.cell(row_idx, 2).text = row_data["Min"]
            dcm_table.cell(row_idx, 3).text = row_data["Max"]

    # Weather Graph
    doc.add_heading("Weather Graph", level=2)
    img_stream = fig_to_image_bytes(report_data['figure'])
    doc.add_picture(img_stream)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def generate_pdf_report(df, report_title, observation, fig, df_dcm=None):
    report_data = build_report_data(df, report_title, observation, fig, df_dcm=df_dcm)

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, report_data['title'], new_x="LMARGIN", new_y="NEXT")

    # Metadata Table
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 8, "Field", border=1)
    pdf.cell(130, 8, "Value", border=1)
    pdf.ln()

    pdf.set_font("Helvetica", size=10)
    for key, value in report_data['metadata']:
        pdf.cell(50, 8, key, border=1)
        pdf.cell(130, 8, str(value), border=1)
        pdf.ln()

    # Column Overview
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Column Overview", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 8, report_data['columns'])

    # Numeric Summary
    if report_data['numeric_summary']:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Numeric Summary", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "B", 10)
        headers = ["Column", "Mean", "Min", "Max"]
        for header in headers:
            pdf.cell(45, 8, header, border=1)
        pdf.ln()

        pdf.set_font("Helvetica", size=10)
        for row_data in report_data['numeric_summary']:
            pdf.cell(45, 8, str(row_data["Column"]), border=1)
            pdf.cell(45, 8, row_data["Mean"], border=1)
            pdf.cell(45, 8, row_data["Min"], border=1)
            pdf.cell(45, 8, row_data["Max"], border=1)
            pdf.ln()

    # DC Meter Summary
    if report_data['dcm_summary']:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "DC Meter Summary", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "B", 10)
        headers = ["Column", "Mean", "Min", "Max"]
        for header in headers:
            pdf.cell(45, 8, header, border=1)
        pdf.ln()

        pdf.set_font("Helvetica", size=10)
        for row_data in report_data['dcm_summary']:
            pdf.cell(45, 8, str(row_data["Column"]), border=1)
            pdf.cell(45, 8, row_data["Mean"], border=1)
            pdf.cell(45, 8, row_data["Min"], border=1)
            pdf.cell(45, 8, row_data["Max"], border=1)
            pdf.ln()

    # Weather Graph
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Weather Graph", new_x="LMARGIN", new_y="NEXT")

    img_stream = fig_to_image_bytes(report_data['figure'])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(img_stream.getvalue())
        temp_image_path = tmp.name

    pdf.image(temp_image_path, w=180)

    os.unlink(temp_image_path)

    return bytes(pdf.output())


# =========================
# SUPABASE-DRIVEN SETTINGS & FETCHERS
# =========================

NUM_SENSORS = 24


def get_forced_sensors() -> set:
    """Best-effort read of which sensor IDs currently have the
    sub-zero cutoff overridden (pi_settings.force_log_sensors, a
    jsonb array of ints). Returns an empty set (i.e. nothing forced —
    normal/safe behavior) if the column/row doesn't exist yet or the
    request fails, so a Supabase hiccup never shows a false 'currently
    forcing' state. Not cached — needs to reflect toggles immediately."""
    try:
        res = (
            supabase.table("pi_settings")
            .select("force_log_sensors")
            .eq("id", 1)
            .execute()
        )
        if res.data:
            raw = res.data[0].get("force_log_sensors") or []
            return {int(s) for s in raw}
    except Exception:
        pass
    return set()


def set_sensor_force(sensor_id: int, forced: bool) -> bool:
    """Best-effort toggle of a single sensor's force-log override.
    Reads the current array, adds/removes sensor_id, writes the full
    array back. Returns True on success so the caller can show an
    error if it didn't actually go through."""
    try:
        current = get_forced_sensors()
        if forced:
            current.add(sensor_id)
        else:
            current.discard(sensor_id)
        supabase.table("pi_settings").update(
            {"force_log_sensors": sorted(current)}
        ).eq("id", 1).execute()
        return True
    except Exception:
        return False


@st.cache_data(ttl=5)
def fetch_latest_readings(limit=50):
    response = (
        supabase.table("sensor_readings")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = response.data
    if not rows:
        return pd.DataFrame()

    # flatten the jsonb "readings" column into normal columns
    flat_rows = []
    for r in rows:
        flat = {"created_at": r["created_at"], "date": r.get("date"), "time": r.get("time")}
        flat.update(r.get("readings") or {})
        flat_rows.append(flat)

    df_live = pd.DataFrame(flat_rows)
    df_live = df_live.sort_values("created_at")  # oldest -> newest for plotting
    return df_live


@st.cache_data(ttl=5)
def fetch_recent_alerts(limit=200):
    response = (
        supabase.table("sensor_alerts")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = response.data
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


@st.cache_data(ttl=5)
def fetch_latest_panel_readings(limit=200):
    response = (
        supabase.table("panel_readings")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = response.data
    if not rows:
        return pd.DataFrame()

    df_panel = pd.DataFrame(rows)
    df_panel = df_panel.sort_values("created_at")  # oldest -> newest for plotting
    return df_panel
