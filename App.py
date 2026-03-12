import streamlit as st
import pandas as pd
import joblib

# Set up the page layout
st.set_page_config(page_title="Order Cancellation Analysis", layout="centered")

st.title("📦 E-commerce Order Cancellation Analysis")

# --- CREATE THE TABS ---
tab1, tab2, tab3 = st.tabs(["🤖 Prediction Machine", "📊 Data Exploration", "💬 Project Assistant"])

# ==========================================
# TAB 1: THE PREDICTION MACHINE
# ==========================================
with tab1:
    st.write("Adjust the features below to predict the probability that an order will be cancelled.")
    st.divider()

    st.header("Order Features")
    col1, col2 = st.columns(2)

    with col1:
        order_amount = st.number_input("Order Amount ($)", min_value=0.0, value=120.50)
        customer_tenure = st.number_input("Customer Tenure (Days)", min_value=0, value=45)
        discount_applied = st.checkbox("Discount Applied?")

    with col2:
        shipping_method = st.selectbox("Shipping Method", ["Standard", "Express", "Next-Day"])
        payment_type = st.selectbox("Payment Type", ["Credit Card", "E-Wallet", "Bank Transfer"])
        items_in_cart = st.slider("Number of Items", min_value=1, max_value=20, value=3)

    input_data = pd.DataFrame({
        'order_amount': [order_amount],
        'customer_tenure': [customer_tenure],
        'discount_applied': [1 if discount_applied else 0],
        'shipping_method': [shipping_method],
        'payment_type': [payment_type],
        'items_in_cart': [items_in_cart]
    })

    st.divider()

    if st.button("Predict Cancellation Probability", type="primary"):
        try:
            probability = 0.28 # Dummy value for testing the UI
            
            st.header("Results")
            st.metric(label="Cancellation Probability", value=f"{probability * 100:.1f}%")

            if probability > 0.5:
                st.error("🚨 High Risk: This order is likely to be cancelled.")
            else:
                st.success("✅ Low Risk: This order is likely to be completed.")

        except Exception as e:
            st.error(f"Error making prediction: {e}")

# ==========================================
# TAB 2: DATA EXPLORATION (EDA)
# ==========================================
with tab2:
    st.header("Understanding the Data")
    st.write("Before building the Random Forest model, we analyzed the dataset to identify key factors driving order cancellations.")
    
    try:
        st.image("correlation ofall.jpg", caption="Feature Correlation Matrix")
    except Exception as e:
        st.warning("Upload 'correlation ofall.jpg' to GitHub to see the image here.")
        
    st.info("💡 Key insights and other charts can be added here.")

# ==========================================
# TAB 3: PROJECT CHATBOT
# ==========================================
with tab3:
    st.header("Project Assistant")
    st.write("Ask me anything about the data preparation, model training, or insights from this project!")
    
    # Placeholder for the chat interface
    st.chat_message("assistant").write("Hello! I am the AI assistant for this project. How can I help you?")
