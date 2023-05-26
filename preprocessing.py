import os
from InvertedIndexTable import InvertedIndexTable
import pickle
import nltk
from nltk.corpus import words

if __name__ == '__main__':

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('words')

    inverted_index_table = InvertedIndexTable()

    correct_words = words.words()

    if os.path.exists("preprocessing_data\\para_id.pickle"):
        with open("preprocessing_data\\para_id.pickle", "rb") as f:
            para_id = pickle.load(f)
    else:
        para_id = 0

    for type in os.listdir("parsed_json"):
        print("--------------------------------------------")
        print(type)
        for document in os.listdir("parsed_json\\" + type):
            print(document)
            para_id = inverted_index_table.add_document("parsed_json\\" + type + "\\" + document, para_id, type)

            with open("preprocessing_data\\para_id.pickle", "wb") as f:
                pickle.dump(para_id, f)

            os.rename("parsed_json\\" + type + "\\" + document, "done_processing\\" + type + "\\" + document)

    with open("preprocessing_data\\english_words.pickle", "wb") as f:
        pickle.dump(correct_words, f)
