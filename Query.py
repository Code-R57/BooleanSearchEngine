import nltk
import regex
from nltk.stem import PorterStemmer
import os
import json
import pickle
from nltk.corpus import wordnet
from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance
from collections import defaultdict

operators = {
    "binary operators": [
        (r"^OR$", "OR"),
        (r"\\[\d]+", "\\n"),
        (r"^AND$", "AND")
    ],
    "unary operators": [
        (r"\\s\([\w\s]+\)", "\\s"),
        (r"\"[\w\s]+\"", "\"w\""),
        (r"\!\([\w\s]+\)", "!")
    ],
    "wildcard operators": [
        (r"\*[a-zA-Z]+$", "<suffix>"),
        (r"^[a-zA-Z]+\*", "<prefix>"),
        (r"(?<=\*)[\w]+(?=\*)", "<substring>")
    ]
}


def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))


def get_correct_word(word, english_words):
    correct_words = set([(jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))), w) for w in english_words if w[0] == word[0]])
    correct_words_sorted = sorted(correct_words, key=lambda val: val[0])
    corrected_words = [tuple[1] for tuple in correct_words_sorted]
    return corrected_words


def get_precedence(token):
    if regex.match(operators["binary operators"][2][0], token):
        return 4
    elif regex.match(operators["binary operators"][1][0], token):
        return 3
    elif regex.match(operators["binary operators"][0][0], token):
        return 2
    elif token in ["(", ")"]:
        return 5
    else:
        return 0


class QueryProcessing:
    def __init__(self, query_tree, query, tags):
        self.query_tree = query_tree
        self.tags = tags
        self.query = query
        self.recent_words = {}
        self.recent_words["auto"] = {}
        self.recent_words["property"] = {}

        if os.path.exists("preprocessing_data\\stop_words.pickle"):
            with open("preprocessing_data\\stop_words.pickle", "rb") as f:
                self.stopwords = pickle.load(f)

        if os.path.exists("preprocessing_data\\recent_words.pickle"):
            with open("preprocessing_data\\recent_words.pickle", "rb") as f:
                self.recent_words = pickle.load(f)

        if os.path.exists("preprocessing_data\\english_words.pickle"):
            with open("preprocessing_data\\english_words.pickle", "rb") as f:
                self.english_words = pickle.load(f)

    def get_stemmed_token(self, token, stemmer: PorterStemmer):
        token = str(token).lower()
        token = stemmer.stem(token)
        return token

    def merge_union(self, query_list):
        result_list = {}

        if len(query_list) == 0 or len(query_list[0].keys()) == 0:
            return result_list

        total_lists = len(query_list)
        for i in range(total_lists):
            if "<STOP_WORD>" in list(query_list[i].keys())[0]:
                del query_list[i]
                i = i - 1

        key_term = "NOT" if "NOT" in list(query_list[0].keys())[0] else "UNION"
        result_list[key_term] = {}
        processed = []
        for query_list_element in query_list:
            for term, dicti in query_list_element.items():
                for para_id, data in dicti.items():
                    if para_id != 'paragraph_freq':
                        if para_id in processed:

                            result_list[key_term][para_id]["position"] = result_list[key_term][para_id]["position"] + \
                                                                         data["position"]
                            result_list[key_term][para_id]["position"] = list(
                                set(result_list[key_term][para_id]["position"]))
                            result_list[key_term][para_id]["position"].sort()
                            result_list[key_term][para_id]["freq"] = len(result_list[key_term][para_id]["position"])
                        else:
                            processed.append(para_id)
                            result_list[key_term][para_id] = data.copy()
                            if result_list[key_term][para_id].get("sentence_id"):
                                del result_list[key_term][para_id]["sentence_id"]

        result_list[key_term] = dict(sorted(result_list[key_term].items(), key=lambda x: x[0]))
        result_list[key_term]["paragraph_freq"] = len(result_list[key_term])

        return result_list

    def merge_intersect(self, query_list):
        result_list = {}
        if len(query_list) == 0 or len(query_list[0].keys()) == 0:
            return result_list

        key_term = "NOT" if "NOT" in list(query_list[0].keys())[0] else "INTERSECT"

        result_list[key_term] = {}

        pointers = [0] * len(query_list)

        for para_id in list(query_list[0].values())[0].keys():
            if para_id == "paragraph_freq":
                break

            for i in range(1, len(query_list)):
                word_para_id = list(list(query_list[i].values())[0].keys())

                while pointers[i] < len(word_para_id) and word_para_id[pointers[i]] < para_id:
                    pointers[i] += 1

                if pointers[i] == len(word_para_id):
                    break

                if word_para_id[pointers[i]] == para_id:
                    continue
                else:
                    break
            else:
                positions = []

                for j in range(0, len(query_list)):
                    positions = positions + list(query_list[j].values())[0][para_id]["position"]

                positions = sorted(list(set(positions)))

                result_list[key_term][para_id] = {}
                result_list[key_term][para_id]["position"] = positions
                result_list[key_term][para_id]["freq"] = len(positions)

        result_list[key_term]["paragraph_freq"] = len(result_list[key_term].keys())
        return result_list

    def merge_minus(self, query1, query2):
        result_list = {}
        position = 0

        if len(query2.keys()) == 0:
            return query1

        result_list[list(query1.keys())[0]] = {}

        for key1 in query1[list(query1.keys())[0]].keys():
            if "NOT" in key1:
                break
            if key1 == "paragraph_freq":
                continue

            while position < len(query2[list(query2.keys())[0]].keys()) and list(query2[list(query2.keys())[0]].keys())[
                position] < key1:
                position += 1

            if position == len(query2[list(query2.keys())[0]].keys()):
                result_list[list(query1.keys())[0]][key1] = query1[list(query1.keys())[0]][key1]

            if list(query2[list(query2.keys())[0]].keys())[position] == key1:
                continue
            else:
                result_list[list(query1.keys())[0]][key1] = query1[list(query1.keys())[0]][key1].copy()

        result_list[list(query1.keys())[0]]["paragraph_freq"] = len(result_list[list(query1.keys())[0]].keys())
        return result_list

    def basic_query_retrieval(self, query_words, is_synonym=False):
        stemmer = PorterStemmer()
        results = []


        for query_word in query_words.split():
                # get_synonyms(query_word)
            sections_list = {}
            for i, tag in enumerate(self.tags.split(" ")):
                if query_word in self.stopwords or any(char.isdigit() for char in query_word):
                    sections_list["<STOP_WORD> " + query_word] = {} # = { "paragraph_freq" : 0 }
                else:
                    outer_path = os.getcwd()
                    inner_path = outer_path + "/inverted_index/" + tag

                    stemmed_word = self.get_stemmed_token(query_word, stemmer)
                    ch = stemmed_word[0]

                    if self.recent_words[tag].get(stemmed_word):
                        sections_list[stemmed_word] = self.recent_words[tag][stemmed_word].copy()
                        sections_list[stemmed_word]["paragraph_freq"] = len(sections_list[stemmed_word])
                        results.append(sections_list.copy())
                    else:
                        file_path = inner_path + "/" + ch + ".json"
                        if os.path.exists(file_path):
                            with open(file_path) as f:
                                data = json.load(f)

                                if data.get(stemmed_word):
                                    sections_list[stemmed_word] = data[stemmed_word].copy()
                                    sections_list[stemmed_word]["paragraph_freq"] = len(sections_list[stemmed_word])
                                    self.recent_words[tag][stemmed_word] = data[stemmed_word].copy()
                                    results.append(sections_list.copy())
                                else:
                                    if not is_synonym:
                                        if not sections_list:
                                            original_tag = self.tags
                                            if query_word.lower() not in self.english_words:
                                                self.tags = tag
                                                for ind, corrected_word in enumerate(get_correct_word(query_word.lower(), self.english_words)):
                                                    res = self.basic_query_retrieval(corrected_word, True)
                                                    sections_list[stemmed_word] = res[list(res.keys())[0]].copy()

                                                    if not sections_list:
                                                        if ind == 10:
                                                            break
                                                        continue
                                                    else:
                                                        results.append(sections_list.copy())
                                                        break
                                                else:
                                                    del sections_list[stemmed_word]
                                                    sections_list["<STOP_WORD>"] = {}
                                            else:
                                                self.tags = tag
                                                for ind, synonym in enumerate(get_synonyms(query_word.lower())):
                                                    res = self.basic_query_retrieval(synonym, True)
                                                    sections_list[stemmed_word] = res[list(res.keys())[0]].copy()

                                                    if not sections_list:
                                                        if ind == 10:
                                                            break
                                                        continue
                                                    else:
                                                        results.append(sections_list.copy())
                                                        break
                                                else:
                                                    del sections_list[stemmed_word]
                                                    sections_list["<STOP_WORD>"] = {}
                                            self.tags = original_tag
            if not sections_list and is_synonym:
                return {}
        if len(results) == 0:
            return {"<STOP_WORD>": {}}
        elif len(results) == 1:
            results[0][stemmed_word]["paragraph_freq"] = len(results[0][stemmed_word].keys())
            return results[0]
        else:
            result_list = {}

            key_term = list(results[0].keys())[0]
            result_list[key_term] = {}
            pointers = [0] * len(results)
            while pointers[0] != len(results[0][key_term].keys()) and pointers[1] != len(results[1][key_term].keys()):
                if list(results[0][key_term].keys())[pointers[0]] == 'paragraph_Freq':
                    pointers[0] = pointers[0] + 1
                    break
                if list(results[1][key_term].keys())[pointers[1]] == 'paragraph_Freq':
                    pointers[1] = pointers[1] + 1
                    break

                if list(results[0][key_term].keys())[pointers[0]] < list(results[1][key_term].keys())[pointers[1]]:
                    result_list[key_term][list(results[0][key_term].keys())[pointers[0]]] = \
                    list(results[0][key_term].values())[pointers[0]]
                    pointers[0] = pointers[0] + 1
                else:
                    result_list[key_term][list(results[1][key_term].keys())[pointers[1]]] = \
                    list(results[1][key_term].values())[pointers[1]]
                    pointers[1] = pointers[1] + 1

            while pointers[0] != len(results[0][key_term].keys()):
                result_list[key_term][list(results[0][key_term].keys())[pointers[0]]] = \
                list(results[0][key_term].values())[pointers[0]]
                pointers[0] = pointers[0] + 1

            while pointers[1] != len(results[1][key_term].keys()):
                result_list[key_term][list(results[1][key_term].keys())[pointers[1]]] = \
                list(results[1][key_term].values())[pointers[1]]
                pointers[1] = pointers[1] + 1

            result_list[key_term]['paragraph_freq'] = len(result_list[key_term].keys())
            return result_list

    def wildcard_query_retrieval(self, query_type, query_affixes):
        words_list = {}

        for query_affix, _ in query_affixes.items():
            file_path = "affixes/" + query_type.replace("<", "").replace(">", "") + "/"
            path = file_path
            if query_type == "<suffix>":
                path = path + query_affix[-1] + ".json"
            else:
                path = path + query_affix[0] + ".json"

            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)

                    if data.get(query_affix):
                        words_list[query_affix] = data[query_affix]
                
        return words_list

    def not_query(self, sub_query):
        result_list = {}
        results = self.merge_union(sub_query)

        for key, result in results.items():
            if "<STOP_WORD>" not in key:
                result_list["NOT " + key] = result
        return result_list

    def phrase_query(self, phrase):
        result_list = {}
        pointers = [0] * len(phrase)

        first_list_ind = 0

        while "<STOP_WORD>" in list(phrase[first_list_ind].keys())[0]:
            first_list_ind += 1

        result_list["PHRASE " + list(phrase[first_list_ind].keys())[0]] = {}

        if first_list_ind == len(phrase):
            result_list = {}
            result_list["<STOP_WORD>"] = {}
            return result_list

        for para_id in list(phrase[first_list_ind].values())[0].keys():
            if para_id == "paragraph_freq":
                break

            word_ind_list = list(phrase[first_list_ind].values())[0][para_id]["position"]

            for i in range(first_list_ind + 1, len(phrase)):
                if "<STOP_WORD>" in list(phrase[i].keys())[0]:
                    continue

                word_para_id = list(list(phrase[i].values())[0].keys())

                while pointers[i] < len(word_para_id) - 1 and word_para_id[pointers[i]] < para_id:
                    pointers[i] += 1

                if pointers[i] == len(word_para_id) - 1:
                    break

                if word_para_id[pointers[i]] == para_id:
                    flag = False
                    word_pos_position = 0

                    for word_ind in list(phrase[i].values())[0][para_id]["position"]:
                        while word_pos_position < len(word_ind_list) and word_ind_list[
                            word_pos_position] < word_ind - i + first_list_ind:
                            word_pos_position += 1

                        if word_pos_position == len(word_ind_list):
                            break

                        if word_ind_list[word_pos_position] == word_ind - i + first_list_ind:
                            flag = True
                            break

                    if flag:
                        continue
                    else:
                        break
                else:
                    break
            else:
                positions = []

                word_pos_position = 0

                for word_ind in list(phrase[len(phrase) - 1].values())[0][para_id]["position"]:
                    while word_pos_position < len(word_ind_list) and word_ind_list[word_pos_position] < word_ind - i + first_list_ind:
                        word_pos_position += 1

                    if word_pos_position == len(word_ind_list):
                        break

                    if word_ind_list[word_pos_position] == word_ind - i + first_list_ind:
                        positions.append([word_ind - i + first_list_ind, word_ind])
                    else:
                        break

                if len(positions) > 0:
                    result_list["PHRASE " + list(phrase[first_list_ind].keys())[0]][para_id] = {}
                    result_list["PHRASE " + list(phrase[first_list_ind].keys())[0]][para_id]["position"] = positions
                    result_list["PHRASE " + list(phrase[first_list_ind].keys())[0]][para_id]["freq"] = len(positions)
        result_list["PHRASE " + list(phrase[first_list_ind].keys())[0]]["paragraph_freq"] = len(
            result_list["PHRASE " + list(phrase[first_list_ind].keys())[0]].keys())
        return result_list

    def sentence_query(self, word_list):
        result_list = {}
        pointers = [0] * len(word_list)

        first_list_ind = 0

        while "<STOP_WORD>" in list(word_list[first_list_ind].keys())[0]:
            first_list_ind += 1

        result_list["SENTENCE " + list(word_list[first_list_ind].keys())[0]] = {}

        if first_list_ind == len(word_list):
            result_list = {}
            result_list["<STOP_WORD>"] = {}
            return result_list

        for para_id in list(word_list[first_list_ind].values())[0].keys():
            if para_id == "paragraph_freq":
                break

            sentence_ind_list = list(word_list[first_list_ind].values())[0][para_id]["sentence_id"]

            for i in range(first_list_ind + 1, len(word_list)):
                if "<STOP_WORD>" in list(word_list[i].keys())[0]:
                    continue

                word_para_id = list(list(word_list[i].values())[0].keys())

                while pointers[i] < len(word_para_id) - 1 and word_para_id[pointers[i]] < para_id:
                    pointers[i] += 1

                if pointers[i] == len(word_para_id) - 1:
                    break

                if word_para_id[pointers[i]] == para_id:
                    flag = False

                    sentence_pos_position = 0

                    for sentence_ind in list(word_list[i].values())[0][para_id]["sentence_id"]:
                        while sentence_pos_position < len(sentence_ind_list) and sentence_ind_list[sentence_pos_position] < sentence_ind:
                            sentence_pos_position += 1

                        if sentence_pos_position == len(sentence_ind_list):
                            break

                        if sentence_ind_list[sentence_pos_position] == sentence_ind:
                            flag = True
                            break

                    if flag:
                        continue
                    else:
                        break
                else:
                    break
            else:
                positions = []
                sentence_id = []

                sentence_pos_position = [0] * len(word_list)

                for sentence_ind in list(word_list[first_list_ind].values())[0][para_id]["sentence_id"]:
                    for j in range(first_list_ind + 1, len(word_list)):
                        while sentence_pos_position[j] < len(list(word_list[j].values())[0][para_id]["sentence_id"]) and \
                                list(word_list[j].values())[0][para_id]["sentence_id"][
                                    sentence_pos_position[j]] < sentence_ind:
                            sentence_pos_position[j] += 1

                        if sentence_pos_position[j] == len(list(word_list[j].values())[0][para_id]["sentence_id"]):
                            break

                        if list(word_list[j].values())[0][para_id]["sentence_id"][
                            sentence_pos_position[j]] == sentence_ind:
                            continue
                        else:
                            break
                    else:
                        sentence_id.append(sentence_ind)

                if len(sentence_id) > 0:
                    positions = []

                    for j in range(0, len(word_list)):
                        positions = positions + list(word_list[j].values())[0][para_id]["position"]

                    positions = sorted(list(set(positions)))

                    result_list["SENTENCE " + list(word_list[first_list_ind].keys())[0]][para_id] = {}
                    result_list["SENTENCE " + list(word_list[first_list_ind].keys())[0]][para_id][
                        "position"] = positions
                    result_list["SENTENCE " + list(word_list[first_list_ind].keys())[0]][para_id]["freq"] = len(
                        positions)
                    result_list["SENTENCE " + list(word_list[first_list_ind].keys())[0]][para_id][
                        "sentence_id"] = sentence_id
        result_list["SENTENCE " + list(word_list[first_list_ind].keys())[0]]["paragraph_freq"] = len(
            result_list["SENTENCE " + list(word_list[first_list_ind].keys())[0]].keys())

        return result_list

    def word_distance_query(self, word_list, operator):
        result_list = {}
        pointers = [0] * len(word_list)
        number = int(regex.findall(r"[\d]+", operator)[0]) + 1

        result_list["WORD_DISTANCE " + list(word_list[0].keys())[0]] = {}

        is_not = [False] * len(word_list)

        if "<STOP_WORD>" in list(word_list[0].keys())[0] and "<STOP_WORD>" in list(word_list[1].keys())[0]:
            result_list["<STOP_WORD>"] = {}
            return result_list
        elif "<STOP_WORD>" in list(word_list[0].keys())[0]:
            return word_list[1]
        elif "<STOP_WORD>" in list(word_list[1].keys())[0]:
            return word_list[0]

        for i in range(len(word_list)):
            if "NOT" in list(word_list[i].keys())[0]:
                is_not[i] = True

        for para_id in list(word_list[0].values())[0].keys():
            if para_id == "paragraph_freq":
                break

            word_ind_list = list(word_list[0].values())[0][para_id]["position"]
            start_ind_list = word_ind_list
            end_ind_list = word_ind_list

            if isinstance(word_ind_list[0], list):
                start_ind_list = [sublist[0] for sublist in word_ind_list]
                end_ind_list = [sublist[-1] for sublist in word_ind_list]

            for i in range(1, len(word_list)):
                word_para_id = list(list(word_list[i].values())[0].keys())

                while pointers[i] < len(word_para_id) - 1 and word_para_id[pointers[i]] < para_id:
                    pointers[i] += 1

                if pointers[i] == len(word_para_id) - 1:
                    break

                if word_para_id[pointers[i]] == para_id:
                    flag = False

                    word_pos_position = 0

                    for word_ind in list(word_list[i].values())[0][para_id]["position"]:
                        word_id = word_ind
                        if isinstance(word_ind, list):
                            word_id = word_ind[0]

                        while word_pos_position < len(end_ind_list) and end_ind_list[
                            word_pos_position] < word_id - number:
                            word_pos_position += 1

                        if word_pos_position == len(end_ind_list):
                            break

                        if end_ind_list[word_pos_position] == word_id - number:
                            flag = True
                            break

                    if flag and not is_not[0] and not is_not[i]:
                        continue
                    else:
                        break
                else:
                    break
            else:
                positions = []

                word_pos_position = 0

                for ind, word_ind in enumerate(list(word_list[len(word_list) - 1].values())[0][para_id]["position"]):
                    word_id = word_ind
                    end_id = word_ind
                    if isinstance(word_ind, list):
                        word_id = word_ind[0]
                        end_id = word_ind[1]
                    while word_pos_position < len(end_ind_list) and end_ind_list[word_pos_position] < word_id - number:
                        word_pos_position += 1

                    if word_pos_position == len(end_ind_list):
                        break

                    if end_ind_list[word_pos_position] == word_id - number and not is_not[0] and not is_not[
                        len(word_list) - 1]:
                        positions.append([start_ind_list[word_pos_position], end_id])

                if is_not[0] or is_not[len(word_list) - 1]:
                    for ind, word_ind in enumerate(list(word_list[len(word_list) - 1].values())[0][para_id]):
                        word_id = word_ind
                        end_id = word_ind
                        if isinstance(word_ind, list):
                            word_id = word_ind[0]
                            end_id = word_ind[1]
                        for ind0, word_ind0 in enumerate(end_ind_list):
                            if word_ind0 == word_id - number:
                                continue
                            else:
                                positions.append([start_ind_list[ind0], end_id])

                if len(positions) > 0:
                    result_list["WORD_DISTANCE " + list(word_list[0].keys())[0]][para_id] = {}
                    result_list["WORD_DISTANCE " + list(word_list[0].keys())[0]][para_id]["position"] = sorted(
                        positions)
                    result_list["WORD_DISTANCE " + list(word_list[0].keys())[0]][para_id]["freq"] = len(positions)

        result_list["WORD_DISTANCE " + list(word_list[0].keys())[0]]["paragraph_freq"] = len(
            result_list["WORD_DISTANCE " + list(word_list[0].keys())[0]].keys())
        return result_list

    def wildcard_intersect(self, wildcard_list):
        result_list = {}
        pointers = [0] * len(wildcard_list)

        possible_words = []

        if len(wildcard_list) == 0 or not wildcard_list[0]:
            return {"<STOP_WORD>": {}}

        for word in list(wildcard_list[0].values())[0]:
            for i in range(1, len(wildcard_list)):
                while pointers[i] < len(list(wildcard_list[i].values())[0]) and list(wildcard_list[i].values())[0][
                    pointers[i]] < word:
                    pointers[i] += 1

                if pointers[i] == len(list(wildcard_list[i].values())[0]):
                    break

                if list(wildcard_list[i].values())[0][pointers[i]] == word:
                    continue
                else:
                    break
            else:
                possible_words.append(word)

        words = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(self.query)
        temp_words = possible_words.copy()
        
        for temp_word in temp_words:
            for ch in temp_word:
                if ch.isdigit():
                    possible_words.remove(temp_word)
                    break
        
        if len(possible_words) == 0:
            return {"<STOP_WORD>": {}}

        ngrams_list = []
        for word_list in words:
            for n in range(1, len(word_list) + 1):
                ngrams_list += ngrams(word_list, n)

        freq_dict = defaultdict(int)
        for ngram in ngrams_list:
            if ngram in possible_words:
                freq_dict[ngram] += 1

        best_match_ordered = sorted(freq_dict, key=freq_dict.get, reverse=True)
        candidate_words = [ngram[-1] for ngram in best_match_ordered]
        filtered_words = [word for word in candidate_words if word not in self.query and word in possible_words]

        if len(filtered_words) > 0:
            return self.basic_query_retrieval(filtered_words[0])
        
        return self.basic_query_retrieval(possible_words[0])

    def and_query(self, query_list):
        not_list = []
        normal_list = []
        
        for query in query_list:
            if "NOT" in list(query.keys())[0]:
                not_list.append(query.copy())
            elif "<STOP_WORD>" in list(query.keys())[0]:
                continue
            else:
                normal_list.append(query.copy())

        return self.merge_minus(self.merge_intersect(normal_list).copy(), self.merge_union(not_list).copy())

    def or_query(self, query_list):
        not_list = []
        normal_list = []

        for query in query_list:
            if "NOT" in list(query.keys())[0]:
                not_list.append(query)
            elif "<STOP_WORD>" in list(query.keys())[0]:
                continue
            else:
                normal_list.append(query)

        return self.merge_union(normal_list)

    def process_query(self, query_tree):
        result_list = []

        if isinstance(query_tree, list):
            for sub_queries in query_tree:
                result_list.append(self.process_query(sub_queries))
        else:
            for operator, sub_queries in query_tree.items():
                if len(sub_queries) == 0:
                    return self.basic_query_retrieval(operator)
                elif "*" in operator:
                    wildcard_list = []

                    for wildcard_tree in query_tree[operator]:
                        for affix_type, query_affixes in wildcard_tree.items():
                            for query_affix in query_affixes:
                                wildcard_list.append(self.wildcard_query_retrieval(affix_type, query_affix).copy())

                    return self.wildcard_intersect(wildcard_list)
                elif operator == "!":
                    result_list = self.process_query(sub_queries)
                    return self.not_query(result_list)
                elif operator == "\"w\"":
                    result_list = self.process_query(sub_queries)
                    return self.phrase_query(result_list)
                elif operator == "\\s":
                    result_list = self.process_query(sub_queries)
                    return self.sentence_query(result_list)
                elif operator == "AND":
                    result_list = self.process_query(sub_queries)
                    return self.and_query(result_list)
                elif operator == "OR":
                    result_list = self.process_query(sub_queries)
                    return self.or_query(result_list)
                elif regex.match(operators["binary operators"][1][0], operator):
                    result_list = self.process_query(sub_queries)
                    return self.word_distance_query(result_list, operator)
                else:
                    return result_list
        return result_list


class Query:
    def __init__(self, query):
        self.query = query
        self.query_tree = {}

    def process_query(self):
        operator_stack = []
        processed_queue = []

        tokens = nltk.regexp_tokenize(self.query, r"\\s\([\w\s]+\)|\"[\w\s]+\"|\!\([\w\s]+\)|[\w\*]+|\(|\)|\\[\d]+")

        for token in tokens:
            if not regex.match(r"^AND$|^OR$|\\[\d]+|\(|\)", token):
                processed_queue.append(token)
            elif token in ["(", ")"]:
                if token == "(":
                    operator_stack.append(token)
                else:
                    while operator_stack and operator_stack[-1] != "(":
                        processed_queue.append(operator_stack.pop())
                    if operator_stack and operator_stack[-1] == "(":
                        operator_stack.pop()
            else:
                while operator_stack and operator_stack[-1] != "(" and get_precedence(token) <= get_precedence(
                        operator_stack[-1]):
                    processed_queue.append(operator_stack.pop())
                operator_stack.append(token)

        while operator_stack:
            processed_queue.append(operator_stack.pop())

        return processed_queue

    def build_for_wildcard_operator(self, token):
        node = {token: []}

        for (pattern, type) in operators["wildcard operators"]:
            sub_queries = [{str(x.group().replace("*", "")).lower(): []} for x in regex.finditer(pattern, token)]

            child_node = {type: sub_queries}
            node[token].append(child_node)

        return node

    def build_for_unary_operator(self, token, operator):
        token = token.replace("\\s", "")
        sub_queries = regex.findall(r"[\w]+", token)

        node = {operator: []}

        for sub_query in sub_queries:
            child_node = {sub_query: []}
            node[operator].append(child_node)

        return node

    def build_for_binary_operator(self):
        tokens = self.process_query()
        stack = []

        for token in tokens:
            if token not in ["(", ")", "AND", "OR"] and not regex.match(r"\\[\d]+", token):

                sub_queries = {}

                is_processed = False

                for (pattern, type) in operators["unary operators"]:
                    if regex.match(pattern, token):
                        sub_queries = self.build_for_unary_operator(token, type)
                        is_processed = True
                        break

                if '*' in token:
                    sub_queries = self.build_for_wildcard_operator(token)
                    is_processed = True

                if not is_processed:
                    sub_queries = {token: []}

                node = sub_queries
                stack.append(node)
            elif token in ["AND", "OR"] or regex.match(r"\\[\d]+", token):
                right = stack.pop()
                left = stack.pop()
                node = {token: [left, right]}
                stack.append(node)

        stack[0] = self.optimize_parse_tree(stack[0])

        return stack[0]

    def flatten_and_operands(self, node):
        if isinstance(node, dict) and node.get('AND'):
            and_children = []
            for child in node['AND']:
                and_children.extend(self.flatten_and_operands(child))
            return and_children
        else:
            return [node]

    def optimize_parse_tree(self, node):
        for key in node.keys():
            if isinstance(node[key], list):
                for i in range(len(node[key])):
                    node[key][i] = self.optimize_parse_tree(node[key][i])
            elif isinstance(node[key], dict):
                node[key] = self.optimize_parse_tree(node[key])

        if isinstance(node, dict) and node.get('AND'):
            and_children = []
            for child in node['AND']:
                and_children.extend(self.flatten_and_operands(child))
            node['AND'] = and_children
        return node

    def build_query_tree(self):
        self.query_tree = self.build_for_binary_operator()

    def get_result(self, tag):
        query_processor = QueryProcessing(self.query_tree, self.query, tag)
        return query_processor.process_query(query_processor.query_tree)

