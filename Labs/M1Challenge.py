#import libs
import os
import openai
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv

#load env
load_dotenv()

#instantiate OpenAI client
client = openai.OpenAI()

def get_dataset_path():
    path = os.path.join(os.getcwd(),'data','customer_reviews.csv')
    return path

@st.cache_data
def classify_review(text):
    response = client.responses.create(
        model='gpt-4o',
        input=[
            {'role':'user', 'content':f'Classify the following product review as Positive, Negative or Neutral: {text}. If you have cant find classify the provided text into one of the 3 categories, use Neutral as default.'}
        ],
        temperature=0,
        max_output_tokens=100
    )
    return response.output[0].content[0].text

#main code
st.title('GenAI Sentiment Analysis App')
if 'missing_data_ingested' not in st.session_state:
    st.session_state['missing_data_ingested'] = True

col1, col2 = st.columns(2)

with col1:
    if st.button('💻 Ingest Dataset'):
        try:
            csv_path = get_dataset_path()
            st.session_state['df'] = pd.read_csv(csv_path)
            st.session_state['df']['SUMMARY'] = st.session_state['df']['SUMMARY'].replace('',np.nan)
            st.session_state['missing_data_ingested'] = False
            st.success('Dataset has been ingested successfully!')
        except:
            st.error('Oops! Something went wront when ingesting the dataset :(')
with col2:
    if st.button('💡 Classify Product Reviews', disabled=st.session_state['missing_data_ingested']):
        try:
            with st.spinner('AI is classifying product reviews...'):
                st.session_state['df']['REVIEW_CLASSIFICATION'] = np.where(
                    st.session_state['df']['SUMMARY'].isna(),
                    'Neutral',
                    st.session_state['df']['SUMMARY'].apply(classify_review)
                )
                st.success('Product Reviews have been classified into Neutral, Positive or Negative!')

        except:
            st.error('Oops! Something went wrong when classifying the product reviews :(')

if 'df' in st.session_state:
    st.subheader('🔎 Filter by Product')
    product = st.selectbox("Choose a Product",['All'] + list(st.session_state['df']['PRODUCT'].unique()))
    st.subheader('Dataset Preview')
    if product == 'All':
        filtered_df = st.session_state['df']
    else:
        filtered_df = st.session_state['df'][st.session_state['df']['PRODUCT'] == product]
    st.dataframe(filtered_df)
    if 'REVIEW_CLASSIFICATION' in st.session_state['df'].columns:
        st.subheader('📊 Sentiment Analysis by Product')
        grouped_df = filtered_df[filtered_df['REVIEW_CLASSIFICATION'].isin(['Neutral','Positive','Negative'])].groupby(['REVIEW_CLASSIFICATION'])['SUMMARY'].count()
        df_for_chart = grouped_df.reset_index().rename(columns={'SUMMARY': 'Count'})

        domain_ = ['Positive', 'Negative', 'Neutral']
        range_ = ['#4CAF50', '#F44336', '#FFC107']

        chart = alt.Chart(df_for_chart).mark_bar().encode(
            x=alt.X('REVIEW_CLASSIFICATION:N', title='Type of Classification', sort=domain_),
            y=alt.Y('Count:Q', title='Number of Reviews'),
            color=alt.Color(
                'REVIEW_CLASSIFICATION:N',
                scale=alt.Scale(domain=domain_, range=range_),
                legend=None
            ),
            tooltip=['REVIEW_CLASSIFICATION', 'Count']
        ).properties(
            title="Review Classification Breakdown"
        )

        st.altair_chart(chart, width='stretch')