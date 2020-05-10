"""
defines the schema of the table PARAGRAPH, QUESTION and ANSWER stored in the database
"""

# paragraph schema
PARAGRAPH = {
    'id': "INTEGER",  # paragraph id: from 1 - 20239
    'article': 'INTEGER',  # article id: from 1 - 477
    'context': 'TEXT',  # content of paragraph
    'is_train': 'BOOLEAN'  # if it's training data
}

# question schema
QUESTION = {
    'id': 'TEXT PRIMARY KEY',  # question id: unique string
    'question': 'TEXT',  # content of question
    'is_impossible': 'BOOLEAN',  # if it has answers
    'paragraph': 'INTEGER',  # paragraph id
    'article': 'INTEGER',  # article id
    'is_train': 'INTEGER'  # 1 if it's training data else 0
}

# answer schema
ANSWER = {
    'id': 'TEXT PRIMARY KEY',  # answer: <question_id>_< 1 to length of all possible answers>
    'question_id': 'TEXT',  # question id
    'answer': 'TEXT',  # content of answer
    'answer_start': 'INTEGER',  # index of the first character of answer in the paragraph
}
