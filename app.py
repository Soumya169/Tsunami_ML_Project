import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("tsunami_model.pkl", "rb"))
data = pd.read_csv("tsunami_data_clean.csv")

st.set_page_config(
    page_title="Tsunami Risk Analysis & Prediction",
    page_icon="🌊",
    layout="wide"
)

st.sidebar.title("🌊 Tsunami Dashboard")
page = st.sidebar.radio("Go to", ["Overview", "Data Exploration", "Visualizations", "Prediction"])

if page == "Overview":
    st.title("🌊 Global Earthquake – Tsunami Risk Analysis")
    st.write("""
    This app analyzes global earthquake events (2001–2022) and predicts
    whether a tsunami is likely based on event parameters.
    """)

    st.subheader("Dataset Info")
    st.write(f"**Rows:** {data.shape[0]}  |  **Columns:** {data.shape[1]}")
    st.write("**Columns:**", list(data.columns))

elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    st.write("### Sample Data")
    st.dataframe(data.head())

    st.write("### Summary Statistics")
    st.write(data.describe())

    st.write("### Tsunami vs Non-Tsunami Count")
    st.bar_chart(data["tsunami"].value_counts())

elif page == "Visualizations":
    st.title("📈 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Magnitude Distribution")
        st.bar_chart(np.histogram(data["magnitude"], bins=20)[0])

        st.write("Year-wise Earthquake Count")
        year_counts = data["Year"].value_counts().sort_index()
        st.line_chart(year_counts)

    with col2:
        st.write("Depth Distribution")
        st.bar_chart(np.histogram(data["depth"], bins=20)[0])

        st.write("Monthly Earthquake Count")
        month_counts = data["Month"].value_counts().sort_index()
        st.line_chart(month_counts)

elif page == "Prediction":
    st.title("🤖 Tsunami Risk Prediction")

    st.write("Enter earthquake parameters to predict tsunami risk:")

    col1, col2 = st.columns(2)

    with col1:
        magnitude = st.slider("Magnitude", 6.0, 9.5, 7.0)
        cdi = st.slider("CDI (Community Intensity)", 0, 9, 4)
        mmi = st.slider("MMI (Modified Mercalli Intensity)", 1, 9, 6)
        sig = st.slider("Significance (SIG)", 650, 3000, 870)
        nst = st.slider("Number of Stations (NST)", 0, 950, 200)
        dmin = st.slider("Epicentral Distance (DMIN)", 0.0, 20.0, 1.0)

    with col2:
        gap = st.slider("GAP", 0.0, 240.0, 25.0)
        depth = st.slider("Depth (km)", 0.0, 700.0, 70.0)
        latitude = st.slider("Latitude", -70.0, 75.0, 0.0)
        longitude = st.slider("Longitude", -180.0, 180.0, 50.0)
        year = st.slider("Year", 2001, 2022, 2015)
        month = st.slider("Month", 1, 12, 6)

    if st.button("🔍 Predict Tsunami Risk"):
        # order must match X.columns in training
        input_data = np.array([[magnitude, cdi, mmi, sig, nst,
                                dmin, gap, depth, latitude,
                                longitude, year, month]])

        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.success(f"⚠️ Tsunami likely. Model probability: {proba:.2f}")
        else:
            st.info(f"✅ Tsunami unlikely. Model probability: {proba:.2f}")

    st.markdown("---")
    st.caption("Model: GradientBoostingClassifier (scikit-learn)")
