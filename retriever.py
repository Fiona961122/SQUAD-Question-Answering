"""
defines the Retriever Class whose function is to find the most relevant 5 articles given a question
"""

import sqlite3
from collections import defaultdict

from database import get_data

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from joblib import load, dump
import numpy as np
from numpy.linalg import norm
import random

vectorizer_path = "data/tfidf-vectorizer.jbl"
article_vector_path = "data/article_vec.jbl"
question_pred_path = "data/ques_pred.jbl"
unigram_retriever_path = "data/unigram_retriever.jbl"
unigram_paragraph_retriever_path = "data/unigram_paragraph_retriever.jbl"

class Retriever():

    article = None  # dict of {article_id: list of paragraphs}
    training_questions = None # list of all training questions
    dev_questions = None  # list of (question, article_id) of all dev questions

    def __init__(self, db):
        self.article = self.get_article(db)
        self.training_questions = get_data(db, "question", "question", "WHERE is_train = 1")
        self.dev_questions = get_data(db, "question, article", "question", "WHERE is_train=0")
        self.vectorizer = self.train_vectorizer()
        self.article_vectors = self.get_article_vectors()
        self.dev_question_predictions = self.get_question_prediction()


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

    def train_vectorizer(self):
        try:
            return load(vectorizer_path)

        except IOError:
            all_texts = self.get_all_texts()
            vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
            vectorizer.fit(all_texts)
            dump(vectorizer, vectorizer_path)
            return vectorizer

    def get_all_texts(self):
        output = []

        # get all paras from articles
        for para_list in self.article.values():
            output.extend(para_list)

        # get all questions
        for q in self.training_questions:
            output.append(q[0])

        return output

    def get_article_vectors(self):
        try:
            return load(article_vector_path)

        except IOError:
            output = {}
            for index, para_list in self.article.items():
                output[index] = self.vectorizer.transform([' '.join(para_list)]).A[0]

            dump(output, article_vector_path)

            return output

    def get_question_prediction(self):
        """
        get article predictions using dev questions
        Returns: list of (question_index, (5 top similar article indices))

        """
        try:
            return load(question_pred_path)
        except IOError:
            output = []
            correct = 0

            for q_index, (q_text, answer) in enumerate(self.dev_questions):
                print("currently on question %d/%d..." % (q_index, len(self.dev_questions)))
                q_vec = self.vectorizer.transform([q_text]).A[0]
                predictions = self.get_five_most_similar(q_vec)
                output.append((q_index, predictions))

                if answer in predictions:
                    print("Got question %d correct!" % (q_index))
                    correct+=1

            print("dev testing complete. Accuracy: %.3f" % (correct/len(self.dev_questions)))

            dump(output, question_pred_path)

            return output

    def get_train_question_text_by_index(self, i):
        return self.training_questions[i][0]

    def get_five_most_similar(self, ques_v):
        # get sorted list of cosine similarities and indices
        candidates = [(self.get_cosine_sim_from_vecs(ques_v, article_v), index)
                      for index, article_v in self.article_vectors.items()]

        candidates.sort(key=lambda x:x[0], reverse=True)

        # return indices of first 5 most similar articles
        return tuple([index for _, index in candidates[:5]])

    def get_cosine_sim_from_vecs(self, vec1, vec2):
        """
        computes cosine similarity between two vectors
        """
        return np.dot(vec1, vec2)/(norm(vec1)*norm(vec2))  # return cosine similarity

    def get_top_five_articles_from_question(self, q_text):
        q_vec = self.vectorizer.transform([q_text]).A[0]
        predictions = self.get_five_most_similar(q_vec)

        print("Top 5 most relevant articles for question \"%s\":" % q_text)

        for i, index in enumerate(predictions):
            print("Top %d, Article %d: %s" % (i + 1, index, self.article[index][0][:300]))

        print()

    def get_demo_from_dev_ques_index(self, dev_index):
        r.get_top_five_articles_from_question(self.dev_questions[dev_index][0])
        print("The correct article number is: %d\n" % (self.dev_questions[dev_index][1]))

    def calculate_accuracy_by_article_index(self, dev_index):
        question_inds = [i for i, (_, index) in enumerate(self.dev_questions) if index == dev_index]
        correct = 0
        for ind in question_inds:
            q_vec = self.vectorizer.transform([self.dev_questions[ind][0]]).A[0]
            predictions = self.get_five_most_similar(q_vec)
            isCorrect = self.dev_questions[ind][1] in predictions
            print("Question " + str(ind) + ": " + self.dev_questions[ind][0] + ": " + str(isCorrect))

            if isCorrect:
                correct +=1
        print("Accuracy for article %d: %.3f" % (dev_index, (correct/len(question_inds))))

    def calculate_average_cosine_similarity_to_other_articles(self, art_ind):
        sum = 0
        for i in [x for x in range(1, len(self.article) + 1) if x != art_ind]:
            sum += self.get_cosine_sim_from_vecs(self.article_vectors[i], self.article_vectors[art_ind])

        return sum/(len(self.article) - 1)


if __name__ == "__main__":
    print("Initiating retriever... Might take 5 mins on first run, and about a few seconds after everything has been saved.")
    db = sqlite3.connect("squad.sqlite")
    r = Retriever(db)
    #r.get_top_five_articles_from_question(r.dev_questions[0][0])

    print("Demo from class: ")
    print("Successful retrievals: ")
    r.get_top_five_articles_from_question("What legitimate dynasty came before the Yuan?")
    r.get_top_five_articles_from_question("What planet seemed to buck Newton's gravitational laws?")
    print("Unsuccessful retrievals: ")
    r.get_top_five_articles_from_question("Who was elected in 1859?")
    r.get_top_five_articles_from_question("What is treated with medical chambers?")

    print("\n\nHere\'s 10 randomly generated examples: ")
    for x in range(1, 11):
        print("Demo #%d:" % x)
        demo_index = random.randint(1, 11874)
        r.get_demo_from_dev_ques_index(demo_index)
