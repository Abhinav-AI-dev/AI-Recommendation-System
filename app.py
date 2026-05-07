import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Shopping Recommendation System",
    layout="wide"
    initial_sidebar_state = "expanded"
)

# =====================================================
# HIDE STREAMLIT DEFAULT UI
# =====================================================
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #00ADB5;
    color: white;
    font-size: 16px;
}

.product-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1E293B;
    margin-bottom: 20px;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: gray;
    margin-bottom: 30px;
}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class="title">
🛍️ Smart AI Shopping Assistant
</div>

<div class="subtitle">
Personalized Product Recommendations using Machine Learning
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3081/3081559.png",
    width=120
)

st.sidebar.title("AI Shop")

page = st.sidebar.selectbox(
    "📌 Navigation",
    [
        "Home",
        "Customer Segmentation",
        "Recommendations",
        "Analytics Dashboard"
    ]
)

# =====================================================
# LOAD DATA
# =====================================================
try:
    data = pd.read_csv("Mall_Customers.csv")
    products = pd.read_csv("products.csv")

except:
    st.error("Dataset files not found.")
    st.stop()

# =====================================================
# FEATURE SELECTION
# =====================================================
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# =====================================================
# SCALING
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# ELBOW METHOD
# =====================================================
wcss = []

for i in range(1, 11):

    km = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)

    wcss.append(km.inertia_)

# =====================================================
# MODEL TRAINING
# =====================================================
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

data['Segment'] = kmeans.fit_predict(X_scaled)

# =====================================================
# CLUSTER CENTERS
# =====================================================
centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

# =====================================================
# SEGMENT LOGIC
# =====================================================
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

# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================
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

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":

    st.header("🏠 Welcome to AI Shop")

    st.write("""
    This platform uses Artificial Intelligence and Machine Learning
    to analyze customer behavior and generate personalized product
    recommendations.
    """)

    st.subheader("📂 Customer Dataset Preview")

    st.dataframe(data.head())

# =====================================================
# CUSTOMER SEGMENTATION PAGE
# =====================================================
elif page == "Customer Segmentation":

    st.header("📊 Customer Segmentation")

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        data['Annual Income (k$)'],
        data['Spending Score (1-100)'],
        c=data['Segment'],
        cmap='viridis',
        s=80
    )

    ax.scatter(
        centers[:, 1],
        centers[:, 2],
        c='red',
        s=250,
        marker='X',
        label='Centroids'
    )

    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")

    ax.set_title(
        "K-Means Customer Segmentation"
    )

    ax.legend()

    st.pyplot(fig)

    st.subheader("📈 Elbow Method")

    fig2, ax2 = plt.subplots()

    ax2.plot(
        range(1, 11),
        wcss,
        marker='o'
    )

    ax2.set_xlabel("Clusters")
    ax2.set_ylabel("WCSS")

    st.pyplot(fig2)

# =====================================================
# RECOMMENDATION PAGE
# =====================================================
elif page == "Recommendations":

    st.header("🛍️ Personalized Recommendations")

    st.sidebar.subheader("Enter Customer Details")

    age = st.sidebar.slider(
        "Age",
        18,
        70,
        25
    )

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

    if st.sidebar.button(
        "Generate Recommendation"
    ):

        new_customer = np.array([
            [age, income, spending]
        ])

        scaled_customer = scaler.transform(
            new_customer
        )

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

        for index, row in recommended_products.iterrows():

            with st.container():

                col1, col2 = st.columns([1,3])

                with col1:

                    st.image(
                        "https://via.placeholder.com/150",
                        width=130
                    )

                with col2:

                    st.markdown(
                        f"### {row['Product']}"
                    )

                    st.write(
                        f"📂 Category: {row['Category']}"
                    )

                    st.write(
                        f"💰 Price: ${row['Price']}"
                    )

                    st.write(
                        f"⭐ Rating: {row['Rating']}"
                    )

                    st.button(
                        f"Buy Now - {row['Product']}"
                    )

                st.markdown("---")

# =====================================================
# ANALYTICS DASHBOARD
# =====================================================
elif page == "Analytics Dashboard":

    st.header("📈 Analytics Dashboard")

    st.subheader(
        "Customer Segment Distribution"
    )

    st.bar_chart(
        data['Segment'].value_counts()
    )

    st.subheader(
        "Income vs Spending Analysis"
    )

    fig3, ax3 = plt.subplots(figsize=(10,5))

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

    st.subheader("📊 Dataset Statistics")

    st.dataframe(data.describe())

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.caption(
    "🚀 AI-Powered Shopping Recommendation Website"
)
