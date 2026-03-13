import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai # Add this line!
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
# ==========================================
# TAB 3: PROJECT CHATBOT
# ==========================================
with tab3:
    st.header("💬 Project Assistant")
    st.write("Ask me anything about the data preparation, model training, or insights from this project!")
    
    # 1. Securely configure the Gemini API key
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 2. Define the bot's persona and rules
    project_rules = """
    You are an AI assistant for a data science project created by a second-year Artificial Intelligence student at FPT University. 
    The project predicts e-commerce order cancellation probabilities using a Random Forest model. 
    The dataset features include order amount, customer tenure, discount application, shipping method, payment type, and item count.
    
    Your rules:
    - Answer questions strictly related to this e-commerce data, Random Forest models, data cleaning (like One-Hot Encoding), and Exploratory Data Analysis.
    - If a user asks about unrelated topics (e.g., coding help, general history, weather), politely decline and state that you can only answer questions about the order cancellation project.
    - Keep answers concise, professional, and educational.
    - For now the content of the project is empty, so you can only tell them to wait for more information to be added.
    """
    
    # Initialize the model with the system instructions
    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=project_rules)
    
    # 3. Initialize chat memory in Streamlit's session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        # Add a friendly greeting from the bot
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I am the AI assistant for this e-commerce prediction project. What would you like to know about this project?"
        })

    # 4. Display the chat history on the screen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. Handle new user input
    prompt = st.chat_input("Ask about the model or data...")
    if prompt:
        
        # Display the user's message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Add user message to memory
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate the bot's response
        with st.chat_message("assistant"):
            # We need to pass the previous messages to Gemini so it has context
            # We convert Streamlit's dictionary format into a string format Gemini easily reads
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            
            response = model.generate_content(chat_history)
            st.markdown(response.text)
            
        # Add bot's response to memory
        st.session_state.messages.append({"role": "assistant", "content": response.text})
