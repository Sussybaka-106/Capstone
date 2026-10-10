"""
Streamlit Investigator Dashboard for Public Procurement Collusion Analytics.
Compliant with Stage 17 of the Project Blueprint.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Page Configuration
st.set_page_config(
    page_title="Procurement Collusion Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


@st.cache_data
def load_all_data():
    queue_path = os.path.join(DATA_DIR, "investigation_queue.csv")
    bids_path = os.path.join(DATA_DIR, "clean_bids.csv")
    vendors_path = os.path.join(DATA_DIR, "clean_vendors.csv")
    cobids_path = os.path.join(DATA_DIR, "co_bidding_pairs.csv")
    comm_path = os.path.join(DATA_DIR, "vendor_communities.csv")

    df_queue = pd.read_csv(queue_path) if os.path.exists(queue_path) else pd.DataFrame()
    df_bids = pd.read_csv(bids_path) if os.path.exists(bids_path) else pd.DataFrame()
    df_vendors = pd.read_csv(vendors_path) if os.path.exists(vendors_path) else pd.DataFrame()
    df_cobids = pd.read_csv(cobids_path) if os.path.exists(cobids_path) else pd.DataFrame()
    df_comm = pd.read_csv(comm_path) if os.path.exists(comm_path) else pd.DataFrame()

    return df_queue, df_bids, df_vendors, df_cobids, df_comm


df_queue, df_bids, df_vendors, df_cobids, df_comm = load_all_data()

# Sidebar Navigation
st.sidebar.title(" Collusion Analytics")
st.sidebar.markdown("**System Blueprint Stage 17**")
st.sidebar.markdown("Explainable Anomaly Scoring & Bid Graph Analytics")

if df_queue.empty:
    st.error("No processed data found. Please run the training pipeline first: `python -m src.train`")
    st.stop()

# Header Metrics
st.title("Public Procurement Collusion Analytics Platform")
st.markdown(
    "> **Objective**: Identify unusual bidding patterns, prioritize cases for human investigation, "
    "and provide explainable forensic rationale for each flagged tender."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive KPIs",
    "Investigation Queue",
    "Case Inspector & SHAP",
    "Vendor Network Graph",
    "Entity Resolution"
])

# ----------------- TAB 1: EXECUTIVE KPIS -----------------
with tab1:
    st.subheader("System Overview & Anomaly Indicators")

    total_tenders = len(df_queue)
    high_risk_cases = (df_queue["ml_anomaly_prob"] >= 0.70).sum()
    moderate_risk = ((df_queue["ml_anomaly_prob"] >= 0.40) & (df_queue["ml_anomaly_prob"] < 0.70)).sum()
    total_vendors = len(df_vendors)
    aliases_identified = df_vendors["Is_Alias"].sum() if "Is_Alias" in df_vendors.columns else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tenders Analyzed", f"{total_tenders:,}")
    col2.metric("High Risk Flags", f"{high_risk_cases:,}", delta="Urgent", delta_color="inverse")
    col3.metric("Moderate Risk", f"{moderate_risk:,}")
    col4.metric("Active Vendors", f"{total_vendors:,}")
    col5.metric("Entity Aliases Resolved", f"{aliases_identified:,}")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("##### Anomaly Risk Probability Distribution")
        fig_hist = px.histogram(
            df_queue,
            x="ml_anomaly_prob",
            nbins=25,
            color_discrete_sequence=["#e74c3c"],
            title="Distribution of ML Anomaly Scores (0 to 1)",
            labels={"ml_anomaly_prob": "Collusion Risk Probability"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_chart2:
        st.markdown("##### Flagged Cases by Department")
        top_dept = df_queue[df_queue["ml_anomaly_prob"] >= 0.50]["Department"].value_counts().reset_index()
        top_dept.columns = ["Department", "Suspicious Tenders"]
        fig_dept = px.bar(
            top_dept,
            x="Suspicious Tenders",
            y="Department",
            orientation="h",
            color="Suspicious Tenders",
            color_continuous_scale="Reds",
            title="Departments with Highest Frequency of Flagged Tenders",
        )
        st.plotly_chart(fig_dept, use_container_width=True)

# ----------------- TAB 2: INVESTIGATION QUEUE -----------------
with tab2:
    st.subheader("Prioritized Case Queue for Auditors")
    st.caption("Cases sorted by ML Priority Rank (Rank 1 = Highest Collusion Suspicion)")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        min_p = st.slider("Minimum Risk Probability", 0.0, 1.0, 0.50, 0.05)
    with col_f2:
        dept_filter = st.selectbox("Filter Department", ["All"] + sorted(df_queue["Department"].dropna().unique().tolist()))
    with col_f3:
        collusion_filter = st.selectbox("Ground-Truth Type Filter", ["All"] + sorted(df_queue["Collusion_Type"].dropna().unique().tolist()))

    filtered_queue = df_queue[df_queue["ml_anomaly_prob"] >= min_p]
    if dept_filter != "All":
        filtered_queue = filtered_queue[filtered_queue["Department"] == dept_filter]
    if collusion_filter != "All":
        filtered_queue = filtered_queue[filtered_queue["Collusion_Type"] == collusion_filter]

    display_cols = [
        "priority_rank",
        "Tender_ID",
        "Department",
        "Category",
        "Estimated_Value",
        "Winning_Vendor_ID",
        "winning_bid_ratio",
        "winner_to_second_gap",
        "max_pair_cobids",
        "ml_anomaly_prob",
        "baseline_anomaly_score",
        "shap_explanation",
    ]

    st.dataframe(
        filtered_queue[display_cols].style.format({
            "Estimated_Value": "₹{:,.0f}",
            "winning_bid_ratio": "{:.2%}",
            "winner_to_second_gap": "{:.2%}",
            "ml_anomaly_prob": "{:.3f}",
            "baseline_anomaly_score": "{:.0f}",
        }),
        use_container_width=True,
        height=450,
    )

    st.download_button(
        label="Export Investigation Queue (CSV)",
        data=filtered_queue.to_csv(index=False),
        file_name="investigation_queue_export.csv",
        mime="text/csv",
    )

# ----------------- TAB 3: CASE INSPECTOR & SHAP -----------------
with tab3:
    st.subheader("Deep Case Forensic Inspector")
    st.caption("Select any tender to inspect submitted bids and review the model's explainability breakdown.")

    selected_tender_id = st.selectbox(
        "Select Tender ID to Inspect",
        df_queue["Tender_ID"].tolist(),
        index=0,
    )

    case_row = df_queue[df_queue["Tender_ID"] == selected_tender_id].iloc[0]

    # Forensic Overview Card
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    col_c1.metric("Priority Rank", f"#{case_row['priority_rank']}")
    col_c2.metric("ML Anomaly Probability", f"{case_row['ml_anomaly_prob']*100:.1f}%")
    col_c3.metric("Baseline Red Flag Score", f"{case_row['baseline_anomaly_score']:.0f} / 100")
    col_c4.metric("Winning Vendor", f"{case_row['Winning_Vendor_ID']}")

    st.info(f"**Forensic Rationale**: {case_row['shap_explanation']}")

    col_detail1, col_detail2 = st.columns([3, 2])

    with col_detail1:
        st.markdown("##### Submitted Bids for this Tender")
        t_bids = df_bids[df_bids["Tender_ID"] == selected_tender_id].sort_values("Rank")
        st.dataframe(
            t_bids[["Rank", "Vendor_ID", "Canonical_Vendor_ID", "Bid_Amount", "Winner"]].style.format({
                "Bid_Amount": "₹{:,.2f}"
            }),
            use_container_width=True,
        )

    with col_detail2:
        st.markdown("##### Key Anomaly Metrics vs Norms")
        metrics_df = pd.DataFrame([
            {"Metric": "Winner to 2nd Gap", "Case Value": f"{case_row['winner_to_second_gap']*100:.1f}%", "Market Median": "3.5%"},
            {"Metric": "Winning Bid Ratio", "Case Value": f"{case_row['winning_bid_ratio']*100:.1f}%", "Market Median": "96.2%"},
            {"Metric": "Max Co-Bids with Partner", "Case Value": f"{case_row['max_pair_cobids']}", "Market Median": "2.0"},
            {"Metric": "Bid Count", "Case Value": f"{case_row['bid_count']}", "Market Median": "5.0"},
        ])
        st.table(metrics_df)

# ----------------- TAB 4: VENDOR NETWORK GRAPH -----------------
with tab4:
    st.subheader("Vendor Co-Bidding Network Graph (Stage 10 & 11)")
    st.caption("Visualizing repeated bidding cliques. Thick edges represent pairs bidding together in many tenders.")

    min_edge_cobids = st.slider("Filter Minimum Co-Bidding Tenders", 2, 20, 5)
    filtered_pairs = df_cobids[df_cobids["co_bidding_count"] >= min_edge_cobids]

    if filtered_pairs.empty:
        st.warning("No vendor pairs found meeting this co-bidding threshold.")
    else:
        # Build NetworkX subgraph for visualization
        G = nx.Graph()
        for _, r in filtered_pairs.iterrows():
            G.add_edge(r["Vendor_A"], r["Vendor_B"], weight=r["co_bidding_count"])

        pos = nx.spring_layout(G, seed=42, k=0.5)

        # Plotly Graph construction
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color="#888"),
            hoverinfo="none",
            mode="lines"
        )

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            degree = G.degree(node)
            node_text.append(f"Vendor: {node}<br>Co-bidding Connections: {degree}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[n if G.degree(n) > 3 else "" for n in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale="YlOrRd",
                color=[G.degree(n) for n in G.nodes()],
                size=14,
                colorbar=dict(
                    thickness=15,
                    title=dict(text="Network Degree", side="right"),
                    xanchor="left",
                ),
                line_width=2
            )
        )

        fig_net = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f"Co-Bidding Network Subgraph (Threshold >= {min_edge_cobids} Shared Tenders)",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=550,
            )
        )
        st.plotly_chart(fig_net, use_container_width=True)

# ----------------- TAB 5: ENTITY RESOLUTION -----------------
with tab5:
    st.subheader("Entity Resolution & Shell Company Audit (Stage 7)")
    st.caption("Vendors identified with typographical variations, shell names, or identical registered addresses.")

    alias_records = df_vendors[df_vendors["Cluster_Size"] > 1].sort_values("Canonical_Vendor_ID")

    if alias_records.empty:
        st.info("No alias clusters detected.")
    else:
        st.dataframe(
            alias_records[[
                "Vendor_ID",
                "Vendor_Name",
                "Canonical_Vendor_ID",
                "Is_Alias",
                "Cluster_Size",
                "Address",
                "City",
                "Registration_ID"
            ]],
            use_container_width=True,
        )
