import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Recommendation System",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🧠 AI-Powered Product Recommendation System")
st.subheader("Customer Segmentation using Machine Learning")

# ---------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------
try:
    data = pd.read_csv("Mall_Customers.csv")
    products = pd.read_csv("products.csv")

except:
    st.error("❌ Dataset files not found.")
    st.stop()

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
page = st.sidebar.selectbox(
    "📌 Navigation",
    [
        "Home",
        "Customer Segmentation",
        "Recommendations",
        "Analytics Dashboard"
    ]
)

# ---------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# ---------------------------------------------------
# DATA SCALING
# ---------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# ELBOW METHOD
# ---------------------------------------------------
wcss = []

for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

data['Segment'] = kmeans.fit_predict(X_scaled)

# ---------------------------------------------------
# CENTROIDS
# ---------------------------------------------------
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# ---------------------------------------------------
# SEGMENT NAME FUNCTION
# ---------------------------------------------------
def get_segment_name(segment):

    income = centers[segment][1]
    spending = centers[segment][2]

    if income > 70 and spending > 70:
        return "Premium Customers 💎"

    elif income > 70 and spending < 40:
        return "Careful Buyers 🧠"

    elif income < 40 and spending > 70:
        return "Impulsive Buyers 🔥"

    elif income < 40 and spending < 40:
        return "Budget Customers 💰"

    else:
        return "Average Customers 🙂"

# ---------------------------------------------------
# PRODUCT RECOMMENDATION FUNCTION
# ---------------------------------------------------
def recommend_products(segment_name):

    if "Premium" in segment_name:
        recommended = products[
            products['Category'] == "Luxury"
        ]

    elif "Budget" in segment_name:
        recommended = products[
            products['Category'] == "Budget"
        ]

    elif "Impulsive" in segment_name:
        recommended = products[
            products['Category'] == "Trending"
        ]

    elif "Careful" in segment_name:
        recommended = products[
            products['Category'] == "Value"
        ]

    else:
        recommended = products[
            products['Category'] == "Average"
        ]

    return recommended

# ===================================================
# HOME PAGE
# ===================================================
if page == "Home":

    st.header("🏠 Home")

    st.write("""
    This project uses Machine Learning and K-Means Clustering
    to segment customers based on their behavior and recommend
    suitable products dynamically.
    """)

    st.subheader("📂 Dataset Preview")
    st.dataframe(data.head())

# ===================================================
# CUSTOMER SEGMENTATION PAGE
# ===================================================
elif page == "Customer Segmentation":

    st.header("📊 Customer Segmentation")

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        data['Annual Income (k$)'],
        data['Spending Score (1-100)'],
        c=data['Segment'],
        cmap='viridis',
        s=70
    )

    ax.scatter(
        centers[:, 1],
        centers[:, 2],
        c='red',
        s=250,
        marker='X',
        label='Centroids'
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segmentation using K-Means")

    ax.legend()

    st.pyplot(fig)

    st.subheader("📈 Elbow Method")

    fig2, ax2 = plt.subplots()

    ax2.plot(range(1, 11), wcss, marker='o')

    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("WCSS")

    ax2.set_title("Elbow Method for Optimal K")

    st.pyplot(fig2)

# ===================================================
# RECOMMENDATION PAGE
# ===================================================
elif page == "Recommendations":

    st.header("🛍️ Personalized Recommendations")

    st.sidebar.subheader("Enter Customer Details")

    age = st.sidebar.slider("Age", 18, 70, 25)

    income = st.sidebar.slider(
        "Annual Income (k$)",
        0,
        150,
        50
    )

    spending = st.sidebar.slider(
        "Spending Score",
        1,
        100,
        50
    )

    if st.sidebar.button("Generate Recommendation"):

        new_customer = np.array([
            [age, income, spending]
        ])

        scaled_customer = scaler.transform(new_customer)

        predicted_segment = kmeans.predict(
            scaled_customer
        )[0]

        segment_name = get_segment_name(
            predicted_segment
        )

        st.success(
            f"🎯 Predicted Segment: {segment_name}"
        )

        recommended_products = recommend_products(
            segment_name
        )

        st.subheader("🛒 Recommended Products")

        st.dataframe(recommended_products)

# ===================================================
# ANALYTICS DASHBOARD
# ===================================================
elif page == "Analytics Dashboard":

    st.header("📊 Analytics Dashboard")

    st.subheader("Customer Segment Distribution")

    st.bar_chart(
        data['Segment'].value_counts()
    )

    st.subheader("Income vs Spending")

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    ax3.plot(
        data['Annual Income (k$)'],
        label="Income"
    )

    ax3.plot(
        data['Spending Score (1-100)'],
        label="Spending Score"
    )

    ax3.legend()

    st.pyplot(fig3)

    st.subheader("Dataset Statistics")

    st.dataframe(data.describe())

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.caption(
    "🚀 AI-Powered Recommendation System using Streamlit & Machine Learning"
)
