"""
Transforms raw SQUAD data into an squad.sqlite SQLite database.
"""

import os
import json
import sqlite3

from schema import PARAGRAPH
from schema import QUESTION
from schema import ANSWER

# path of the sqlite file
DBFILE = "squad.sqlite"

# SQL statements
CREATE_TABLE = "CREATE TABLE IF NOT EXISTS {table} ({fields})"
INSERT_ROW = "INSERT INTO {table} ({columns}) VALUES ({values})"
SELECT_ALL = "SELECT {} FROM {}"
SELECT = "SELECT {} FROM {} WHERE {}"


def init():
    """
    Connects initializes a new output SQLite database.
    Returns: connection to the database
    """

    # delete existing file
    if os.path.exists(DBFILE):
        os.remove(DBFILE)

    db = sqlite3.connect(DBFILE)
    # create tables
    create(db, PARAGRAPH, "paragraph")
    create(db, QUESTION, "question")
    create(db, ANSWER, "answer")

    return db


def create(db, table, name):
    """
    Creates a SQLite table.
    Args:
        db: database connection
        table: table schema
        name: table name
    """
    columns = ['{0} {1}'.format(name, ctype) for name, ctype in table.items()]
    create = CREATE_TABLE.format(table=name, fields=", ".join(columns))
    try:
        db.execute(create)
    except Exception as e:
        print(create)
        print("Failed to create table: " + e)


def insert(db, table, name, row):
    """
    Builds and inserts a row.
    Args:
        db: database connection
        table: table object
        name: table name
        row: row to insert
    """

    # Build insert prepared statement
    columns = [name for name, _ in table.items()]
    insert = INSERT_ROW.format(table=name, columns=", ".join(columns), values=("?, " * len(columns))[:-2])

    try:
        db.execute(insert, values(table, row, columns))
    except Exception as ex:
        print("Error inserting row: {}".format(row), ex)


def values(table, row, columns):
    """
    Formats and converts row into database types based on table schema.
    Args:
        table: table schema
        row: row tuple
        columns: column names
    Returns:
        Database schema formatted row tuple
    """

    values = []
    for x, column in enumerate(columns):
        # Get value
        value = row[x]

        if table[column].startswith("INTEGER"):
            values.append(int(value) if value else 0)
        elif table[column] == "BOOLEAN":
            values.append(1 if value else 0)
        elif table[column] == "TEXT":
            # Clean empty text and replace with None
            values.append(value if value and len(value.strip()) > 0 else None)
        else:
            values.append(value)

    return values


def process(data_item, article_id):
    """
    Preocess a single paragraph data
    :param data_item: a dict contains a list of qas and context
    :param article_id: article index
    :param is_train: boolean if it's training data
    :return: a row in paragraph table, a list of rows in question table, a list of rows in answer table
    """
    questions = []
    answers = []
    paragraph = [article_id, data_item['context']]

    for item in data_item['qas']:
        question = [item["id"], item["question"], item['is_impossible']]
        questions.append(question)
        if item['is_impossible']:
            continue
        answer_options = item["answers"]
        answer_set = set()
        for option in answer_options:
            answer_tuple = (option['text'], option['answer_start'])
            answer_set.add(answer_tuple)
        for index, answer_tuple in enumerate(answer_set):
            answer = ["{}_{}".format(item["id"], index+1), item["id"], answer_tuple[0], answer_tuple[1]]
            answers.append(answer)
    return paragraph, questions, answers


def run(indir):
    """
    Main execution method.
    Args:
        indir: input directory
        entryfile: path to entry dates file
    """

    print("Building squad.sqlite from {}".format(indir))

    # Initialize database
    db = init()

    # Paragraph, article indices
    pindex = 0
    aindex = 0

    for root, dirs, files in os.walk(indir):
        for file in files:
            if file.startswith("train"):
                is_train = 1
            else:
                is_train = 0
            with open(os.path.join(root, file)) as f:
                data = json.load(f)
                print("Loading data from {}{}".format(file, is_train))
            for article in data['data']:
                aindex += 1
                for para in article['paragraphs']:
                    pindex += 1
                    paragraph, questions, answers = process(para, aindex)
                    paragraph = [pindex] + paragraph
                    paragraph.append(is_train)
                    insert(db, PARAGRAPH, "paragraph", paragraph)

                    for question in questions:
                        question.append(pindex)
                        question.append(aindex)
                        question.append(is_train)
                        insert(db, QUESTION, "question", question)

                    for answer in answers:
                        insert(db, ANSWER, "answer", answer)

    print("Total articles inserted: {}".format(aindex))
    # print("Total paragraphs inserted: {}".format(pindex))
    paragraphs = get_data(db, "context", "paragraph")
    print("Total paragraphs inserted: {}".format(len(paragraphs)))
    training_questions = get_data(db, "*", "question", "is_train=1")
    dev_questions = get_data(db, "question, article", "question", "is_train=0")
    print("Training questions: {}; Dev questions: {}".format(len(training_questions), len(dev_questions)))


    # Commit changes and close
    db.commit()
    db.close()


def get_data(db, columns, table, condition=None):
    """
    Select data from database
    :param db: database connection
    :param columns: data columns
    :param table: table name
    :param condition: selection conditon
    :return: list of data
    """
    cur = db.cursor()
    if condition is None:
        cur.execute(SELECT_ALL.format(columns, table))
    else:
        cur.execute(SELECT.format(columns, table, condition))
    return cur.fetchall()

if __name__ == "__main__":
    run("data")
