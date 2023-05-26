import os
import json
import pickle
import math


class ParagraphRetrieval:
    def __init__(self, sections_retirieved):
        self.sections_retrieved = sections_retirieved
        self.ranked_sections = {}

    def rank_retrievals(self):
        ranked_sections = {}

        if "<STOP_WORD>" in list(self.sections_retrieved.keys())[0]:
            ranked_sections["<STOP_WORD>"] = {}
            return ranked_sections

        paragraph_freq = list(self.sections_retrieved.values())[0]["paragraph_freq"]
        total_paras = 1

        if os.path.exists("preprocessing_data\\para_id.pickle"):
            with open("preprocessing_data\\para_id.pickle", "rb") as f:
                total_paras = pickle.load(f)
        
        for section in list(self.sections_retrieved.values()):
            for key, value in section.items():
                if key == "paragraph_freq":
                    continue
                else:
                    ranked_sections[key] = math.log(value["freq"] + 1) * math.log((total_paras+1)/(paragraph_freq+1))

        self.ranked_sections = dict(sorted(ranked_sections.items(), key=lambda x: x[1], reverse=True))

        return self.ranked_sections
    
    def retrieve_paragraphs(self):
        retrieval_list = []
        paragraph_list = list(self.ranked_sections.keys())
        section_map = {}

        with open("paragraph_mapping/index.json", 'r') as f:
            if os.path.getsize(f.name) != 0:
                section_map = json.load(f)

        result_size = 10

        for i in range(min(result_size, len(paragraph_list))):
            document_name = ""

            paragraph_list[i] = str(int(paragraph_list[i]) - 1)

            for key, value in section_map.items():
                index_range = key.split(" - ")

                if int(paragraph_list[i]) in range(int(index_range[0]), int(index_range[1]) + 1):
                    document_name = value
                    break

            if document_name != "":
                paragraph_map = {}

                with open("paragraph_mapping\\" + document_name + "\\index.json", 'r') as f:
                    if os.path.getsize(f.name) != 0:
                        paragraph_map = json.load(f)

                section_name = ""

                for key, value in paragraph_map.items():
                    index_range = key.split(" - ")

                    if int(paragraph_list[i]) in range(int(index_range[0]), int(index_range[1]) + 1):
                        section_name = value
                        break

                if section_name != "":
                    sections = {}

                    try:

                        with open("paragraph_mapping\\" + document_name + "\\" + section_name + ".json", 'r') as f:
                            if os.path.getsize(f.name) != 0:
                                sections = json.load(f)

                                section_name = section_name.replace("._.", "->").replace(" ._", "?").replace("._", "\"")

                                data_map = {"document_name": document_name,
                                            "section_name": section_name,
                                            "section": sections[paragraph_list[i]]}
                                retrieval_list.append(data_map)

                    except FileNotFoundError as e:
                        result_size += 1
                        pass
                    except KeyError as k:
                        result_size += 1
                        pass

        return retrieval_list
                