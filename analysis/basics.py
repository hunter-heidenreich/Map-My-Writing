from collections import Counter


class VocabUtils:

    @staticmethod
    def unique_vocab(word_list):
        """
        Returns a count of the words used in a word list

        :param word_list: A list of words used in a document -- List[str]
        :return: A count of the unique words used in a list -- Counter
        """
        cnt = Counter()
        for word in word_list:
            cnt[word] += 1
        return cnt

    @staticmethod
    def top_k_words(word_list, k=10):
        """
        Returns the top k words used in a list

        :param word_list: A list of words used in a document -- List[str]
        :param k: The top words to be shown -- int
        :return: A list of top words and counts -- List[(str, int)]
        """
        return VocabUtils.unique_vocab(word_list).most_common(k)

    @staticmethod
    def aggregate_words_counts(list_of_word_lists):
        """
        Counts the words used across multiple lists (documents)

        :param list_of_word_lists: A list of list of words used in a collection -- List[List[str]]
        :return: A count of the unique words used in a list -- Counter
        """
        cnt = Counter()
        for word_list in list_of_word_lists:
            cnt += VocabUtils.unique_vocab(word_list)
        return cnt

    @staticmethod
    def global_top_k_words(list_of_word_lists, k=10):
        """
        Returns the top k words used in a collection of word lists (documents)

        :param list_of_word_lists: A list of list of words used in a collection -- List[List[str]]
        :param k: The top words to be shown -- int
        :return: A list of top words and counts -- List[(str, int)]
        """
        return VocabUtils.aggregate_words_counts(list_of_word_lists).most_common(k)
