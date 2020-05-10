"""
defines the Retriever Class whose function is to find the most relevant 5 articles given a question
"""

import sqlite3
from collections import defaultdict

from database import get_data

class Retriever():

    article = None  # dict of {article_id: list of paragraphs}
    training_questions = None # list of all training questions
    dev_questions = None  # list of (question, article_id) of all dev questions

    def init(self, db):
        self.article = self.get_article(db)
        self.training_questions = get_data(db, "question", "question", "WHERE is_train = 1")
        self.dev_questions = get_data(db, "question, article", "question", "WHERE is_train=0")


    def get_article(self, db):
        """
        generate article
        :param db: database connection
        :return: a dict with key: article id and value: list of paragraphs
        """
        paragraph = get_data(db, "context, article", "paragraph")
        article = defaultdict(list)
        for p in paragraph:
            article[p[1]].append(p[0])
        return article


if __name__ == "__main__":
    db = sqlite3.connect("squad.sqlite")
    retriever = Retriever()
    retriever.init(db)
