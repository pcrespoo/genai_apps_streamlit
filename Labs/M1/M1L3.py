from dotenv import load_dotenv
import openai
import streamlit as st
import os
import pandas as pd

# @st.cache_data
# def get_response(user_prompt, temperature):
#     response = client.responses.create(
#         model='gpt-4o',
#         input=[
#             {'role':'user', 'content':user_prompt}
#         ],
#         temperature=temperature,
#         max_output_tokens=100
#     )
#     return response

def get_dataset_path():
    csv_path = os.path.join('/home/pcrespo/Documents/estudos/github_repos/genai_apps_streamlit/','data','customer_reviews.csv')
    return csv_path


load_dotenv('../.env')

client = openai.OpenAI()

st.title('Hello, GenAI!')
st.write('This is your first Streamlit app!')

col1, col2 = st.columns(2)

with col1:
    if st.button("Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            st.session_state['df'] = pd.read_csv(csv_path)
            st.success('Dataset loaded successfully!')
        except:
            st.error('Dataset not found. Please check the file path.')

with col2:
    if st.button("Parse Reviews"):
        pass

if 'df' in st.session_state:
    st.subheader('Filter by Product')
    product = st.selectbox('Choose a product',['All products'] + list(st.session_state['df']['PRODUCT'].unique()))
    st.subheader("Dataset preview")
    if product != "All products":
        filtered_df = st.session_state['df'][st.session_state['df']['PRODUCT'] == product]
    else:
        filtered_df = st.session_state['df']
    st.dataframe(filtered_df)

    st.subheader('Average Sentiment Score by Product')
    grouped_df = st.session_state['df'].groupby(['PRODUCT'])['SENTIMENT_SCORE'].mean()
    st.bar_chart(grouped_df)