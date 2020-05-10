"""
defines the Retriever Class whose function is to find the most relevant 5 articles given a question
"""

import sqlite3

from database import get_data

class Retriever():

    paragraph = None  # list of all paragraphs
    training_questions = None # list of all training questions
    dev_questions = None  # list of (question, article_id) of all dev questions

    def init(self, db):
        self.paragraph = get_data(db, "context", "paragraph")
        self.training_questions = get_data(db, "question", "question", "is_train = 1")
        self.dev_questions = get_data(db, "question, article", "question", "is_train=0")


if __name__ == "__main__":
    db = sqlite3.connect("squad.sqlite")
    retriever = Retriever()
    retriever.init()
