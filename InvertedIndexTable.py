import os
import json
import nltk
from nltk.stem import PorterStemmer
from string import ascii_lowercase as alc
import collections
import pickle


class InvertedIndexTable:
    def __init__(self, file=None):
        if file is None:
            self.data = {}
        else:
            self.data = json.load(file)
        self.file = file

    def add_document(self, doc_file, start_para_id, tag):
        doc_data = {}
        with open(doc_file, encoding='utf-8') as f:
            doc_data = json.load(f)

        stemmer = PorterStemmer()
        stop_words = set(nltk.corpus.stopwords.words('english'))

        with open("preprocessing_data\\stop_words.pickle", "wb") as f:
            pickle.dump(stop_words, f)

        title = doc_data["title"]

        section_map = {}

        os.makedirs("paragraph_mapping\\" + title, exist_ok=True)

        if not os.path.exists("paragraph_mapping\\" + title + "\\index.json"):
            open("paragraph_mapping\\" + title + "\\index.json", 'w')
        with open("paragraph_mapping\\" + title + "\\index.json", 'r') as f:
            if os.path.getsize(f.name) != 0:
                section_map = json.load(f)

        para_id = start_para_id

        index_data = {}
        prefix_data = {}
        suffix_data = {}
        substring_data = {}

        for c in alc:
            if not os.path.exists("inverted_index\\" + tag + "\\" + c + ".json"):
                open("inverted_index\\" + tag + "\\" + c + ".json", 'w')
            with open("inverted_index\\" + tag + "\\" + c + ".json", "r") as f:
                if os.path.getsize(f.name) != 0:
                    index_data[c] = json.load(f)
                else:
                    index_data[c] = {}

            if not os.path.exists("affixes\\prefix\\" + c + ".json"):
                open("affixes\\prefix\\" + c + ".json", 'w')
            with open("affixes\\prefix\\" + c + ".json", "r") as f:
                if os.path.getsize(f.name) != 0:
                    prefix_data[c] = json.load(f)
                else:
                    prefix_data[c] = {}

            if not os.path.exists("affixes\\suffix\\" + c + ".json"):
                open("affixes\\suffix\\" + c + ".json", 'w')
            with open("affixes\\suffix\\" + c + ".json", "r") as f:
                if os.path.getsize(f.name) != 0:
                    suffix_data[c] = json.load(f)
                else:
                    suffix_data[c] = {}

            if not os.path.exists("affixes\\substring\\" + c + ".json"):
                open("affixes\\substring\\" + c + ".json", 'w')
            with open("affixes\\substring\\" + c + ".json", "r") as f:
                if os.path.getsize(f.name) != 0:
                    substring_data[c] = json.load(f)
                else:
                    substring_data[c] = {}

        title_map = {}
        if not os.path.exists("paragraph_mapping\\index.json"):
            open("paragraph_mapping\\index.json", 'w')
        with open("paragraph_mapping\\index.json", 'r') as f:
            if os.path.getsize(f.name) != 0:
                title_map = json.load(f)

        for section in doc_data["sections"]:
            section_heading = section["section_heading"].replace("->", "._.").replace("?", " ._").replace("\"", "._").replace(";", "")

            section_para_id = para_id

            paragraph_map = {}

            for paragraph in section["paragraphs"]:
                paragraph_map[para_id] = paragraph
                para_id += 1

                sentence_id = 0
                word_ind = 0

                for sentence in nltk.tokenize.sent_tokenize(paragraph):

                    tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(sentence)
                    grammar_tags = nltk.pos_tag(tokens)

                    for word in tokens:
                        word = str(word).lower().encode('ascii', 'ignore').decode()
                        if word not in stop_words or not any(char.isdigit() for char in word) or len(word) > 1:
                            word = stemmer.stem(word)

                            ch = word[0]

                            if index_data.get(ch) is None:
                                index_data[ch] = {}
                            if index_data[ch].get(word) is None:
                                index_data[ch][word] = {
                                    str(para_id): {
                                        "position": [word_ind],
                                        "sentence_id": [sentence_id],
                                        "freq": 1
                                    }
                                }
                                index_data[ch] = collections.OrderedDict(sorted(index_data[ch].items()))
                            else:
                                if index_data[ch][word].get(str(para_id)) is None:
                                    index_data[ch][word][str(para_id)] = {
                                        "position": [word_ind],
                                        "sentence_id": [sentence_id],
                                        "freq": 1
                                    }

                                    index_data[ch][word] = collections.OrderedDict(sorted(index_data[ch][word].items()))
                                else:
                                    if word_ind not in index_data[ch][word][str(para_id)]["position"]:
                                        index_data[ch][word][str(para_id)]["position"].append(word_ind)
                                    if sentence_id not in index_data[ch][word][str(para_id)]["sentence_id"]:
                                        index_data[ch][word][str(para_id)]["sentence_id"].append(sentence_id)
                                    index_data[ch][word][str(para_id)]["freq"] = index_data[ch][word][str(para_id)][
                                                                                     "freq"] + 1

                                    index_data[ch][word][str(para_id)]["position"] = sorted(
                                        index_data[ch][word][str(para_id)]["position"])
                                    index_data[ch][word][str(para_id)]["sentence_id"] = sorted(
                                        index_data[ch][word][str(para_id)]["sentence_id"])

                            prefixes = [word[:i] for i in range(1, len(word) + 1) if word[:i][0].isalpha()]
                            suffixes = [word[i:] for i in range(0, len(word)) if word[i:][-1].isalpha()]
                            substrings = [word[i:j] for i in range(len(word)) for j in range(i + 1, len(word) + 1) if
                                          word[i:j][0].isalpha()]

                            for prefix in prefixes:
                                c = prefix[0]
                                if prefix_data[c].get(prefix) is None:
                                    prefix_data[c][prefix] = [word]
                                    prefix_data[c] = collections.OrderedDict(sorted(prefix_data[c].items()))
                                else:
                                    if word not in prefix_data[c][prefix]:
                                        prefix_data[c][prefix].append(word)
                                        prefix_data[c][prefix] = sorted(prefix_data[c][prefix])

                            for suffix in suffixes:
                                c = suffix[-1]
                                if suffix_data[c].get(suffix) is None:
                                    suffix_data[c][suffix] = [word]
                                    suffix_data[c] = collections.OrderedDict(sorted(suffix_data[c].items()))
                                else:
                                    if word not in suffix_data[c][suffix]:
                                        suffix_data[c][suffix].append(word)
                                        suffix_data[c][suffix] = sorted(suffix_data[c][suffix])

                            for substring in substrings:
                                c = substring[0]
                                if substring_data[c].get(substring) is None:
                                    substring_data[c][substring] = [word]
                                    substring_data[c] = collections.OrderedDict(sorted(substring_data[c].items()))
                                else:
                                    if word not in substring_data[c][substring]:
                                        substring_data[c][substring].append(word)
                                        substring_data[c][substring] = sorted(substring_data[c][substring])

                            if index_data[ch][word].get("paragraph_freq"):
                                index_data[ch][word]["paragraph_freq"] = index_data[ch][word]["paragraph_freq"] + 1
                            else:
                                index_data[ch][word]["paragraph_freq"] = 1

                        word_ind += 1
                    sentence_id += 1

            with open("paragraph_mapping\\" + title + "\\" + section_heading + ".json", 'w') as f:
                json.dump(paragraph_map, f)

            section_map[str(section_para_id) + " - " + str(para_id - 1)] = section_heading

        title_map[str(start_para_id) + " - " + str(para_id - 1)] = title

        with open("paragraph_mapping\\" + title + "\\index.json", "w") as f:
            json.dump(section_map, f)
        with open("paragraph_mapping\\index.json", "w") as f:
            json.dump(title_map, f)

        index_data[c] = collections.OrderedDict(sorted(index_data[c].items()))

        for c in alc:
            with open("inverted_index\\" + tag + "\\" + c + ".json", "w") as f:
                json.dump(index_data[c], f)
            with open("affixes\\prefix\\" + c + ".json", "w") as f:
                json.dump(prefix_data[c], f)
            with open("affixes\\suffix\\" + c + ".json", "w") as f:
                json.dump(suffix_data[c], f)
            with open("affixes\\substring\\" + c + ".json", "w") as f:
                json.dump(substring_data[c], f)

        return para_id