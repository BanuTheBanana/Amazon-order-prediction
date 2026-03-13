import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Set up the page layout
st.set_page_config(page_title="Order Cancellation Analysis", layout="centered")

st.title("📦 E-commerce Order Cancellation Analysis")

# --- CREATE THE TABS ---
tab1, tab2, tab3 = st.tabs(["🤖 Prediction Machine", "📊 Data Exploration", "💬 Project Assistant"])

# ==========================================
# TAB 1: THE PREDICTION MACHINE
# ==========================================
with tab1:
    st.write("Adjust the features below to predict the probability of a successful delivery.")
    st.divider()

    st.header("Order Features")
    col1, col2 = st.columns(2)

    with col1:
        quantity = st.number_input("Quantity", min_value=1, value=1)
        price = st.number_input("Amount (INR)", min_value=0.0, value=850.0)
        
        size_list = ['Free', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL', '4XL', '5XL', '6XL']
        size_label = st.select_slider("Garment Size", options=size_list, value='M')
        size_encoded = size_list.index(size_label) 

    with col2:
        is_business = st.checkbox("Is Business Buyer (B2B)?")
        promotion_count = st.number_input("Promotions Applied (Count)", min_value=0, value=0)
        service_level = st.selectbox("Service Level", [0, 1, 2]) 

    # THE TRANSLATION LAYER
    final_input = pd.DataFrame({
        'Qty': [quantity],
        'Amount': [price],
        'Size_Int': [size_encoded],
        'Service_Level_Int': [service_level],
        'Promotion_Count': [promotion_count],
        'B2B': [1 if is_business else 0]
    })

    st.divider()

    if st.button("Predict Success Probability", type="primary"):
        try:
            # 1. Load the model and make the prediction
            model = joblib.load("LG_model.joblib")
            probability = model.predict_proba(final_input)[0][1] 
            
            # 2. Display the top-line result
            st.header("Results")
            st.metric(label="Likelihood of Successful Delivery", value=f"{probability * 100:.1f}%")

            if probability >= 0.5:
                st.success("✅ High Confidence: This order is likely to be successfully fulfilled.")
            else:
                st.error("🚨 Low Confidence: This order is at high risk of cancellation.")

            # --- 3. EXPLAINABLE AI (SHAP) SECTION ---
            st.divider()
            st.subheader("🧠 Model Explanation")
            st.write("This waterfall chart shows exactly how each feature pushed the model's decision higher (red) or lower (blue) from the baseline.")
            
            # Create the SHAP Explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values for our specific UI input
            shap_values = explainer(final_input)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 4))
            # Plot the explanation for the single prediction (index 0)
            shap.plots.waterfall(shap_values[0], show=False)
            
            # Render the plot in Streamlit
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error making prediction or generating explanation: {e}")

# ... (Keep your Tab 2 and Tab 3 code exactly the same below this) ...
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
