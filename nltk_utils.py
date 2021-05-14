{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "55e09f6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk\n",
    "from nltk.stem.porter import PorterStemmer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cb21a3f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#initialize stemmer\n",
    "stemmer = PorterStemmer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e2499f61",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Hey', ',', 'how', 'are', 'you', 'doing', 'nowadays', '?']\n"
     ]
    }
   ],
   "source": [
    "#tokenizer\n",
    "def tokenize(sentence):\n",
    "    return nltk.word_tokenize(sentence)\n",
    "\n",
    "#stemmer function\n",
    "def stem(word):\n",
    "    return stemmer.stem(word)\n",
    "\n",
    "#get the bag of words\n",
    "def bag_of_words(tokenized_sentence, all_words):\n",
    "    pass"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
