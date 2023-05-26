# Boolean Search Engine

The program is a search engine that works on boolean queries and is based on automobile and property policy papers. It also has an option to filter search results by automobile or property tags.  
This program is part of an [assignment](/BooleanSearchEngine.pdf) for **CS F469 - Information Retrieval** course of **[BITS Pilani, Hyderabad Campus](https://www.bits-pilani.ac.in/hyderabad/)**.

## Running the Program

1. Install all the Python libraries required: `pip install -r requirements.txt`.
1. Run the GUI using the command: `streamlit run GUI.py`.
1. Search for any term and select the appropriate filter to get search results.

**NOTE:** Please delete the files in `inverted_index` and `affixes` directories and run the pre-processsing step to ensure accurate index and presence of all indexed files for the first run.

## Operators

| Operator Type |  Operator  |      Example        |
|:-------------:|:----------:|:-------------------:|
| Boolean       | AND        | example AND demo    |
| Boolean       | OR         | example OR demo     |
| Boolean       | ! (NOT)    | !(example)          |
| Phrase        | " "        | "example demo"      |
| Sentence      | \\s        | exmaple \\s demo    |
| Proximity     | \\d        | exmaple \\2 demo    |
| Wildcard      | * (pre)    | exm*                |
| Wildcard      | * (suffix) | *ple                |
| Wildcard      | * (sub)    | exm*ple             |
| Complex       |            | example AND !(demo) |

## Preprocessing

1. Add the parsed json file into the respective sub-directory of `parsed_json` directory.
1. Run the command: `python preprocessing.py` or `python3 preprocessing.py`.  

The parsed json file for a document should have the format:

```json
{
    "title" : "demo",
    "sections" : 
    [
        { 
            "section_heading" : "demo",
            "paragraphs":
                [
                    "demo",
                    "demo"
                ]
        },
        {
            "section_heading" : "demo",
            "paragraphs":
                [
                    "demo",
                    "demo"
                ]
        }
            
    ]
}
```

## Group Members

- **[Ritvik](https://github.com/Code-R57)**
- **[Abhinav Tyagi](https://github.com/Abhiinv)**
- **[Abhinav Verma](https://github.com/vermaabhinav363)**
