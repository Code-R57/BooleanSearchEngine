import streamlit as st
from Query import Query
import time
from ParagraphRetrieval import ParagraphRetrieval
import os
import pickle
import nltk
from nltk.corpus import words

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

if not os.path.exists("preprocessing_data\\para_id.pickle"):
    with open("preprocessing_data\\para_id.pickle", "wb") as f:
        para_id = 0
        pickle.dump(para_id, f)

if not os.path.exists("preprocessing_data\\english_words.pickle"):
    with open("preprocessing_data\\english_words.pickle", "wb") as f:
        correct_words = words.words()
        pickle.dump(correct_words, f)

if not os.path.exists("preprocessing_data\\stop_words.pickle"):
    with open("preprocessing_data\\stop_words.pickle", "wb") as f:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        pickle.dump(stop_words, f)

st.title('Search Engine')

query = st.text_input("Enter query")

radio_button = st.radio('Select tag: ',
               ('ALL', 'Automobile', 'Property'))

if st.button('Search'):

    starting_seconds = time.time()

    tag = ""
    if radio_button == 'Automobile':
        tag = "auto"
    elif radio_button == 'Property':
        tag = "property"
    else:
        tag = "property auto"

    processed = Query(query)
    processed.build_query_tree()
    result = processed.get_result(tag)
    result_updated = result.copy()

    for key, value in result_updated.items():
        if len(value) == 0:
            del result[key]
        elif value["paragraph_freq"] == 0:
            del result[key]
    
    if result:
        st.text("Results generated :" + str(len(list(result.values())[0].keys())-1))

        paragraph_retriever = ParagraphRetrieval(result)
        paragraph_retriever.rank_retrievals()
        retrieval_list = paragraph_retriever.retrieve_paragraphs()

    ending_seconds = time.time()
    st.write("Time taken : " + str(round(ending_seconds-starting_seconds, 4)) + " seconds")

    if result:
        for result in retrieval_list:
            try:
                st.header(result["document_name"])
                st.subheader(result["section_name"])
                st.write(result["section"])
                st.write("\n\n\n")
            except:
                pass
    else:
        st.subheader("No Results Found")
