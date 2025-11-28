import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="🌊 Tsunami Risk Analysis & Prediction",
    page_icon="🌊",
    layout="wide"
)

sns.set_style("whitegrid")

# ------------------ LOAD MODEL & DATA ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "tsunami_model.pkl")
data_path = os.path.join(BASE_DIR, "tsunami_data_clean.csv")

with open(model_path, "rb") as f:
    model = pickle.load(f)

data = pd.read_csv(data_path)

# Ensure column order (must match training)
FEATURE_COLUMNS = [
    "magnitude", "cdi", "mmi", "sig", "nst",
    "dmin", "gap", "depth", "latitude",
    "longitude", "Year", "Month"
]

# ------------------ HELPER: BASIC STATS ------------------
total_events = len(data)
total_tsunami = int((data["tsunami"] == 1).sum())
total_no_tsunami = int((data["tsunami"] == 0).sum())
tsunami_ratio = total_tsunami / total_events if total_events > 0 else 0

year_min, year_max = int(data["Year"].min()), int(data["Year"].max())

# ------------------ SIDEBAR NAV ------------------
st.sidebar.title("🌊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Analytics", "Prediction"],
    index=0
)

# ======================================================
# 1️⃣ OVERVIEW PAGE
# ======================================================
if page == "Overview":
    st.title("🌊 Global Earthquake – Tsunami Risk Analysis")
    st.markdown(
        """
        Explore **earthquake events (2001–2022)** and how they relate to **tsunami occurrence**.
        This dashboard lets you:
        
        - 📊 **Understand the dataset** (Earthquake intensity, depth, location, etc.)
        - 🔍 **Analyze tsunami vs non-tsunami patterns**
        - 🤖 **Predict tsunami risk** for a new earthquake event
        
        ---
        """
    )

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", f"{total_events}")
    with col2:
        st.metric("Tsunami Events", f"{total_tsunami}")
    with col3:
        st.metric("Non-Tsunami Events", f"{total_no_tsunami}")
    with col4:
        st.metric("Tsunami Percentage", f"{tsunami_ratio*100:.1f}%")

    st.markdown("### 📈 Tsunami vs Non-Tsunami Count")
    counts = data["tsunami"].value_counts().sort_index()
    tsunami_labels = ["No Tsunami (0)", "Tsunami (1)"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(tsunami_labels, counts.values, edgecolor="black")
    ax.set_ylabel("Number of Events")
    st.pyplot(fig)

    st.markdown("### 🕒 Year-wise Earthquake Activity")
    yearly_count = data["Year"].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(yearly_count.index, yearly_count.values, marker="o")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Number of Earthquakes")
    ax2.set_title("Yearly Earthquake Count")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.markdown("### 🌎 Geographic Distribution (Quick View)")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    scatter = ax3.scatter(
        data["longitude"],
        data["latitude"],
        c=data["tsunami"],
        cmap="coolwarm",
        alpha=0.6,
        s=20
    )
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_title("Earthquake Locations (Red = Tsunami)")
    st.pyplot(fig3)

    st.caption(f"Dataset period: **{year_min}–{year_max}** | Source: Global Earthquake–Tsunami Risk Assessment Dataset")

# ======================================================
# 2️⃣ ANALYTICS PAGE
# ======================================================
elif page == "Analytics":
    st.title("📊 Detailed Analytics & Visualizations")

    st.markdown(
        """
        Below are some key distributions and relationships, similar to your Colab EDA:
        - Histograms with KDE (magnitude, depth, sig, latitude, longitude)
        - Boxplots comparing **tsunami vs non-tsunami**
        - Scatterplots (location and magnitude vs depth)
        """
    )

    # ---------- Distributions ----------
    st.subheader("1. Feature Distributions")

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    sns.histplot(data=data["magnitude"], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Magnitude Distribution")
    axes[0, 0].set_xlabel("Magnitude")

    sns.histplot(data=data["depth"], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Depth Distribution")
    axes[0, 1].set_xlabel("Depth (km)")

    sns.histplot(data=data["sig"], kde=True, ax=axes[0, 2])
    axes[0, 2].set_title("Significance (SIG) Distribution")
    axes[0, 2].set_xlabel("SIG")

    sns.histplot(data=data["latitude"], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Latitude Distribution")
    axes[1, 0].set_xlabel("Latitude")

    sns.histplot(data=data["longitude"], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Longitude Distribution")
    axes[1, 1].set_xlabel("Longitude")

    sns.histplot(data=data["Year"], kde=False, ax=axes[1, 2])
    axes[1, 2].set_title("Earthquakes by Year")
    axes[1, 2].set_xlabel("Year")
    plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    # ---------- Boxplots ----------
    st.subheader("2. Tsunami vs Non-Tsunami (Boxplots)")

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 8))

    sns.boxplot(x="tsunami", y="magnitude", data=data, ax=axes2[0, 0])
    axes2[0, 0].set_title("Magnitude vs Tsunami")

    sns.boxplot(x="tsunami", y="depth", data=data, ax=axes2[0, 1])
    axes2[0, 1].set_title("Depth vs Tsunami")

    sns.boxplot(x="tsunami", y="sig", data=data, ax=axes2[0, 2])
    axes2[0, 2].set_title("SIG vs Tsunami")

    sns.boxplot(x="tsunami", y="longitude", data=data, ax=axes2[1, 0])
    axes2[1, 0].set_title("Longitude vs Tsunami")

    sns.boxplot(x="tsunami", y="latitude", data=data, ax=axes2[1, 1])
    axes2[1, 1].set_title("Latitude vs Tsunami")

    axes2[1, 2].axis("off")  # empty

    plt.tight_layout()
    st.pyplot(fig2)

    # ---------- Scatterplots ----------
    st.subheader("3. Scatterplots")

    fig3, axes3 = plt.subplots(1, 2, figsize=(18, 5))

    sns.scatterplot(
        x="longitude",
        y="latitude",
        hue="tsunami",
        data=data,
        ax=axes3[0],
        alpha=0.7,
        edgecolor="black"
    )
    axes3[0].set_title("Longitude vs Latitude (Colored by Tsunami)")
    axes3[0].set_xlabel("Longitude")
    axes3[0].set_ylabel("Latitude")

    sns.scatterplot(
        x="magnitude",
        y="depth",
        hue="tsunami",
        data=data,
        ax=axes3[1],
        alpha=0.7,
        edgecolor="black"
    )
    axes3[1].set_title("Magnitude vs Depth (Colored by Tsunami)")
    axes3[1].set_xlabel("Magnitude")
    axes3[1].set_ylabel("Depth (km)")

    plt.tight_layout()
    st.pyplot(fig3)

    st.caption("These visualizations are adapted from your Colab EDA for a web-friendly format.")

# ======================================================
# 3️⃣ PREDICTION PAGE
# ======================================================
elif page == "Prediction":
    st.title("🤖 Tsunami Risk Prediction")
    st.markdown(
        """
        Use this page to **simulate a new earthquake event** and estimate whether a tsunami is likely.
        
        - Adjust parameters using sliders  
        - Or type exact values in the numeric fields  
        - Get prediction + probability  
        - See where your event lies in the magnitude distribution
        """
    )

    # compute ranges from data
    def col_min_max(col):
        return float(data[col].min()), float(data[col].max()), float(data[col].mean())

    mag_min, mag_max, mag_mean = col_min_max("magnitude")
    cdi_min, cdi_max, cdi_mean = col_min_max("cdi")
    mmi_min, mmi_max, mmi_mean = col_min_max("mmi")
    sig_min, sig_max, sig_mean = col_min_max("sig")
    nst_min, nst_max, nst_mean = col_min_max("nst")
    dmin_min, dmin_max, dmin_mean = col_min_max("dmin")
    gap_min, gap_max, gap_mean = col_min_max("gap")
    depth_min, depth_max, depth_mean = col_min_max("depth")
    lat_min, lat_max, lat_mean = col_min_max("latitude")
    lon_min, lon_max, lon_mean = col_min_max("longitude")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Earthquake Intensity & Signal")

        mag_slider = st.slider("Magnitude (slider)", mag_min, mag_max, mag_mean, step=0.1)
        magnitude = st.number_input("Magnitude (exact)", mag_min, mag_max, mag_slider, step=0.1)

        cdi_slider = st.slider("CDI (Community Intensity)", int(cdi_min), int(cdi_max), int(cdi_mean), step=1)
        cdi = st.number_input("CDI (exact)", int(cdi_min), int(cdi_max), cdi_slider, step=1)

        mmi_slider = st.slider("MMI (Modified Mercalli)", int(mmi_min), int(mmi_max), int(mmi_mean), step=1)
        mmi = st.number_input("MMI (exact)", int(mmi_min), int(mmi_max), mmi_slider, step=1)

        sig_slider = st.slider("Significance (SIG)", int(sig_min), int(sig_max), int(sig_mean), step=10)
        sig = st.number_input("SIG (exact)", int(sig_min), int(sig_max), sig_slider, step=10)

        nst_slider = st.slider("Number of Stations (NST)", int(nst_min), int(nst_max), int(nst_mean), step=10)
        nst = st.number_input("NST (exact)", int(nst_min), int(nst_max), nst_slider, step=10)

    with col2:
        st.markdown("#### Location & Geometry")

        dmin_slider = st.slider("Epicentral Distance (dmin)", dmin_min, dmin_max, dmin_mean, step=0.1)
        dmin = st.number_input("dmin (exact)", dmin_min, dmin_max, dmin_slider, step=0.1)

        gap_slider = st.slider("Gap", gap_min, gap_max, gap_mean, step=1.0)
        gap = st.number_input("Gap (exact)", gap_min, gap_max, gap_slider, step=1.0)

        depth_slider = st.slider("Depth (km)", depth_min, depth_max, depth_mean, step=1.0)
        depth = st.number_input("Depth (exact)", depth_min, depth_max, depth_slider, step=1.0)

        latitude_slider = st.slider("Latitude", lat_min, lat_max, lat_mean, step=0.5)
        latitude = st.number_input("Latitude (exact)", lat_min, lat_max, latitude_slider, step=0.5)

        longitude_slider = st.slider("Longitude", lon_min, lon_max, lon_mean, step=0.5)
        longitude = st.number_input("Longitude (exact)", lon_min, lon_max, longitude_slider, step=0.5)

        year = st.slider("Year", int(year_min), int(year_max), int(year_max))
        month = st.slider("Month", 1, 12, 6)

    st.markdown("---")

    if st.button("🔍 Predict Tsunami Risk"):
        # order must match training columns
        input_data = np.array([[
            magnitude, cdi, mmi, sig, nst,
            dmin, gap, depth, latitude,
            longitude, year, month
        ]])

        pred = model.predict(input_data)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1]
        else:
            proba = None

        # Result section
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            if pred == 1:
                st.success("⚠️ Tsunami likely for this event.")
            else:
                st.info("✅ Tsunami unlikely for this event.")

        with result_col2:
            if proba is not None:
                st.metric("Model Tsunami Probability", f"{proba:.2f}")
            else:
                st.write("Model does not support probability output.")

        # Simple visualization: Magnitude position
        st.markdown("#### 📊 Where does your magnitude fall compared to all earthquakes?")
        fig_mag, ax_mag = plt.subplots(figsize=(8, 3))
        sns.histplot(data["magnitude"], kde=True, ax=ax_mag)
        ax_mag.axvline(magnitude, color="red", linestyle="--", label="Your Event")
        ax_mag.set_xlabel("Magnitude")
        ax_mag.set_ylabel("Count")
        ax_mag.legend()
        st.pyplot(fig_mag)

        # Location visualization
        st.markdown("#### 🌎 Your Event Location on Map of Earthquakes")
        fig_loc, ax_loc = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            x="longitude",
            y="latitude",
            hue="tsunami",
            data=data,
            ax=ax_loc,
            alpha=0.4,
            s=20
        )
        ax_loc.scatter(
            [longitude], [latitude],
            color="black",
            s=80,
            marker="X",
            label="Your Event"
        )
        ax_loc.set_xlabel("Longitude")
        ax_loc.set_ylabel("Latitude")
        ax_loc.legend()
        st.pyplot(fig_loc)

        st.caption("Prediction based on GradientBoostingClassifier trained on historical data.")
