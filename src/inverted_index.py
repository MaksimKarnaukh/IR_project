import os
import heapq
import struct
from collections import defaultdict
import pickle
import math
import time
from src import variables
from typing import List, Tuple, Dict, BinaryIO, Generator, Any

DEFAULT_BLOCK_SIZE_LIMIT = 10000

class SPIMI:
    """
    SPIMI (Single-Pass In-Memory Indexing) algorithm for creating an inverted index from a token stream.

    Attributes:
        block_size_limit (int): Maximum number of tokens in a block before writing to disk.
        output_dir (str): Directory where the index file(s) are stored.
        block_num (int): Number of blocks written to disk.
        dictionary (Dict[str, Dict[int, float]]): Dictionary of terms and their postings.
        token_count (int): Number of tokens in the current block.
        term_positions (Dict[str, int]): mapping of term positions in the index file.
        doc_lengths (Dict[int, int]): Dictionary of document lengths.
        doc_count (int): Total number of documents.
        idf_values (Dict[str, float]): Dictionary of IDF values for terms.
        inv_index_in_mem (Dict[str, Dict[int, float]]): In-memory inverted index
        documents (List[str]): List of documents.
        document_titles (List[str]): List of document titles.
        doc_lengths_n (Dict[int, float]): Normalized document lengths.
        previous_rocchio_iteration_query (Dict[str, float]): Previous Rocchio iteration query vector.

    Args:
        block_size_limit (int): Maximum number of tokens in a block before writing to disk.
        output_dir (str): Directory where the index file(s) will be stored.
        force_reindex (bool): Indicates if the index should be rebuilt from scratch.
        documents (list): A list of documents (strings).
        document_titles (list): A list of document titles.

    methods:
        create_index(documents: List[str] | Generator[str, Any, None]) -> None : Create the inverted index from a list of documents.
        write_index_entry(term: str, postings: Dict[int, int], f: BinaryIO) -> None : Write a term and its postings list to a file.
        write_block_to_disk(is_single_index_file: bool = False) -> str : Write the current in-memory term dictionary to a block file on disk.
        read_next_term_postings(f: BinaryIO) -> Tuple[str | None, Dict[int, int] | None] : Read the next term and its postings list from a block file.
        merge_two_blocks(block_file1: str, block_file2: str, is_last_merge: bool = False) -> str : Merge two block files into a single block file.
        merge_blocks(block_files: List[str], delete_merged_blocks: bool = True) -> str : Iteratively merge all block files into a final index.
        spimi_invert(token_stream: List[Tuple[str, int]] | Generator[tuple[Any, int], Any, None]) -> str : Perform SPIMI inversion on a token stream to create an inverted index.
        calculate_document_lengths() : Calculate the length of each document vector and store it in self.doc_lengths.
        get_posting_list(term: str) -> Dict[int, int] | None : Retrieve the posting list for a given term.
        save_index_data() : Save index data to disk.
        load_index_data() : Load index data from disk.
        save_mapping(dictionary: Dict[Any, Any], file_name: str) : Save mapping to disk.
        load_mapping(file_name: str) : Load term positions from disk.
        compute_idf(num_occurrences) : Compute the Inverse Document Frequency (IDF) for a term.
        compute_tf(term_count) : Compute the Log Term Frequency for a term.
        compute_document_tfidf(document_as_list: List[str]) -> Dict[str, float] : Compute the TF-IDF scores for a document.
        remove_query_title_from_results(query_title: str, num_results: int, similar_documents_sorted: List[Tuple[int, float]]) -> List[Tuple[int, float] | None] : Remove the query document from the list of most similar documents.
        fast_cosine_score(query_terms: List[str] | Dict[str, float], k: int = 10, query_doc_title: str | None = None) : Compute the top K documents for a query using cosine similarity.
        rocchio_update(query_vector: Dict[str, float], relevant_docs: List[Dict[str, float]], non_relevant_docs: List[Dict[str, float]], alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.25) -> Dict[str, float] : Update the query vector using the Rocchio algorithm.
        mark_as_relevant(list_of_relevant_indices: List[int], top_k_docs: List[Tuple[int, float]]) -> Tuple[List[int], List[int]] : Mark the relevant documents from the top K documents.
        compute_tfidf_scores_for_doc_list(doc_list: List[int], documents: List[str]) -> List[Dict[str, float]] : Compute the TF-IDF scores for a list of documents.
        rocchio_pipeline(list_of_relevant_indices: List[int], top_docs: List[Tuple[int, float]], query: List[str], query_title: str, first_rocchio: bool = False) -> List[Tuple[int, float]] : Perform the Rocchio algorithm pipeline to update the query vector and the top documents.
    """

    def __init__(self, block_size_limit: int = DEFAULT_BLOCK_SIZE_LIMIT, output_dir: str = variables.inverted_index_folder, force_reindex: bool = False, documents: List[str] | None = None, document_titles: List[str] | None = None):
        """
        Initialize the SPIMI (Single-Pass In-Memory Indexing) instance.

        Args:
            block_size_limit (int): Maximum number of tokens in a block before writing to disk.
            output_dir (str): Directory where the index file(s) will be stored.
            force_reindex (bool): Indicates if the index should be rebuilt from scratch.
            documents (list): A list of documents (strings).
            document_titles (list): A list of document titles.
        """
        self.block_size_limit: int = block_size_limit
        self.output_dir: str = output_dir
        self.block_num: int = 0
        self.dictionary: Dict[str, Dict[int, float]] = dict() # {term: {doc_id: freq}}

        self.inv_index_in_mem: Dict[str, Dict[int, float]] = dict()

        self.token_count: int = 0
        self.term_positions: Dict[str, int] = {}  # {term: offset}

        self.doc_lengths: Dict[int, int] = defaultdict(int)  # {doc_id: doc_length}
        self.doc_lengths_n: Dict[int, float] = defaultdict(float)  # {doc_id: normalized_doc_length}
        self.doc_count: int = 0  # total number of documents

        self.idf_values: Dict[str, float] = {}  # {term: idf_value}
        if force_reindex:
            # remove the existing index files
            for file in os.listdir(output_dir):
                if file.endswith(".bin"):
                    os.remove(os.path.join(output_dir, file))
            os.rmdir(output_dir)
        # if index files do not exist, create them
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            self.create_index(documents)
        else:
            self.load_index_data()

        self.documents = documents
        self.document_titles = document_titles
        self.previous_rocchio_iteration_query = None # needed for rocchio pipeline function, to keep track of the previous rocchio iteration query

    def create_index(self, documents: List[str] | Generator[str, Any, None]) -> None:
        """
        Create the inverted index from a list of documents.

        Args:
            documents (list): A list of documents (strings).
        """
        start_time = time.time()
        token_stream_ = self.generate_token_stream(documents)
        print(f"Token stream generated in: {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        final_index_filename_ = self.spimi_invert(token_stream_)
        print(f"Index creation (SPIMI invert) completed in: {time.time() - start_time:.4f} seconds")
        print(f"Final index written to: {final_index_filename_}\n")

    @staticmethod
    def write_index_entry(term: str, postings: Dict[int, float], f: BinaryIO) -> None:
        """
        Write a term and its postings list to a file.

        Args:
            term (str): The term to be written.
            postings (Dict[int, int]): The postings list for the term.
            f (file object): The file to which the term and postings list will be written.
        """
        try:
            term_bytes = term.encode('utf-8')
            postings_bytes = pickle.dumps(dict(postings))
            f.write(struct.pack('I', len(term_bytes)))
            f.write(term_bytes)
            f.write(struct.pack('I', len(postings_bytes)))
            f.write(postings_bytes)
        except Exception as e:
            print(f"Error writing term '{term}' to disk: {e}")

    def write_block_to_disk(self, is_single_index_file: bool = False) -> str:
        """
        Write the current in-memory term dictionary to a block file on disk.

        Args:
            is_single_index_file (bool): Indicates if there is only one index file (no merging).

        Returns:
            str: The filename of the block file.
        """
        block_filename = os.path.join(self.output_dir, f"block_{self.block_num}.bin")
        with open(block_filename, 'wb') as f:
            sorted_terms = sorted(self.dictionary.items())
            for term, postings in sorted_terms:
                if is_single_index_file: # if we have only one index file, and we will not merge, we need to update all necessary variables here
                    for doc_id, tf in postings.items():
                        postings[doc_id] = self.compute_tf(tf)
                    self.term_positions[term] = f.tell()
                    self.idf_values[term] = self.compute_idf(len(postings))
                    self.inv_index_in_mem[term] = postings
                self.write_index_entry(term, postings, f)

        self.block_num += 1
        self.dictionary = dict() # important to reset the dictionary after writing to disk, for the spimi_invert function to work correctly
        self.token_count = 0
        return block_filename

    @staticmethod
    def read_next_term_postings(f: BinaryIO) -> Tuple[str | None, Dict[int, int] | None]:
        """
        Read the next term and its postings list from a block file.

        Args:
            f (file object): The file from which to read the term and postings list.

        Returns:
            tuple: A tuple containing the term (str) and its postings list (dict).
                   Returns (None, None) if the end of the file is reached.
        """
        term_length_data = f.read(4)
        if not term_length_data:
            return None, None
        term_length: int = struct.unpack('I', term_length_data)[0]
        term: str = f.read(term_length).decode('utf-8')
        postings_length: int = struct.unpack('I', f.read(4))[0]
        postings: Dict[int, int] = pickle.loads(f.read(postings_length))
        return term, postings

    def merge_two_blocks(self, block_file1: str, block_file2: str, is_last_merge: bool = False) -> str:
        """
        Merge two block files into a single block file.

        Args:
            block_file1 (str): The filename of the first block file.
            block_file2 (str): The filename of the second block file.
            is_last_merge (bool): Indicates if this is the last merge step.

        Returns:
            str: The filename of the merged block file.
        """
        f1: BinaryIO = open(block_file1, 'rb')
        f2: BinaryIO = open(block_file2, 'rb')

        term1, postings1 = self.read_next_term_postings(f1)
        term2, postings2 = self.read_next_term_postings(f2)

        merged_block_filename = os.path.join(self.output_dir, f"merged_{self.block_num}.bin")
        with open(merged_block_filename, 'wb') as f:
            offset = f.tell()

            while term1 is not None or term2 is not None:
                current_term: str = ""
                current_postings: Dict[int, float] = dict()
                if term1 is not None and (term2 is None or term1 < term2):
                    current_term, current_postings = term1, postings1
                    term1, postings1 = self.read_next_term_postings(f1)
                elif term2 is not None and (term1 is None or term2 < term1):
                    current_term, current_postings = term2, postings2
                    term2, postings2 = self.read_next_term_postings(f2)
                elif term1 is not None and term2 is not None and term1 == term2: # term1 == term2
                    current_term = term1
                    current_postings = self.merge_dicts(postings1, postings2)
                    term1, postings1 = self.read_next_term_postings(f1)
                    term2, postings2 = self.read_next_term_postings(f2)

                if is_last_merge:
                    for doc_id, tf in current_postings.items():
                        current_postings[doc_id] = self.compute_tf(tf)
                    self.term_positions[current_term] = offset
                    self.idf_values[current_term] = self.compute_idf(len(current_postings))
                    self.inv_index_in_mem[current_term] = current_postings

                self.write_index_entry(current_term, current_postings, f)
                offset = f.tell()

        f1.close()
        f2.close()

        self.block_num += 1
        return merged_block_filename

    def merge_blocks(self, block_files: List[str], delete_merged_blocks: bool = True) -> str:
        """
        Iteratively merge all block files into a final index. Here we use a logarithmic merge strategy.

        Args:
            block_files (list): List of block filenames to be merged.
            delete_merged_blocks (bool): Indicates if the (two) original block files should be deleted after merging.

        Returns:
            str: The filename of the final merged index.
        """
        while len(block_files) > 1:
            new_block_files: List[str] = [] # list of filenames of the new blocks

            for i in range(0, len(block_files), 2):
                if i + 1 < len(block_files):
                    merged_block = self.merge_two_blocks(block_files[i], block_files[i + 1], (len(block_files) == 2))
                    new_block_files.append(merged_block)
                    if delete_merged_blocks:
                        os.remove(block_files[i])
                        os.remove(block_files[i + 1])
                else:
                    new_block_files.append(block_files[i])
            block_files = new_block_files

        final_block: str = block_files[0]
        final_index_filename: str = os.path.join(self.output_dir, 'inverted_index.bin')
        os.rename(final_block, final_index_filename)

        return final_index_filename

    def spimi_invert(self, token_stream: List[Tuple[str, int]] | Generator[tuple[Any, int], Any, None]) -> str:
        """
        Perform SPIMI inversion on a token stream to create an inverted index.

        Args:
            token_stream (generator): A generator that yields (term, doc_id) tuples.

        Returns:
            str: The filename of the final inverted index.
        """
        block_files: List[str] = []

        for term, doc_id in token_stream:
            self.doc_count = max(self.doc_count, doc_id + 1) # ¯\_(ツ)_/¯
            if term not in self.dictionary:

                if self.token_count + 1 > self.block_size_limit: # if the block size limit is reached, write the block to disk
                    block_files.append(self.write_block_to_disk())

                self.dictionary[term] = dict()
                self.token_count += 1

            if doc_id not in self.dictionary[term]:
                self.dictionary[term][doc_id] = 0

            self.dictionary[term][doc_id] += 1
            self.doc_lengths[doc_id] += 1

        if self.token_count > 0:
            block_files.append(self.write_block_to_disk(self.block_num == 0))

        final_index_filename: str = self.merge_blocks(block_files)

        self.calculate_document_lengths()
        self.save_index_data()

        return final_index_filename

    def calculate_document_lengths(self):
        """
        Calculate the length of each document vector and store it in self.doc_lengths.
        This is done after the final index is built.

        The formula for the length of a document vector is:
        length(d) = sqrt(sum(tfidf^2))
        """
        with open(os.path.join(self.output_dir, 'inverted_index.bin'), 'rb') as f:
            while True:
                term_length_data = f.read(4)
                if not term_length_data:
                    break
                term_length = struct.unpack('I', term_length_data)[0]
                term = f.read(term_length).decode('utf-8')
                postings_length = struct.unpack('I', f.read(4))[0]
                postings = pickle.loads(f.read(postings_length))

                idf: float = self.idf_values[term]
                for doc_id, tf in postings.items():
                    tfidf: float = tf * idf
                    self.doc_lengths_n[doc_id] += tfidf ** 2

        for doc_id in self.doc_lengths_n:
            self.doc_lengths_n[doc_id] = math.sqrt(self.doc_lengths_n[doc_id])

    def get_posting_list(self, term: str) -> Dict[int, int] | None:
        """
        Retrieve the posting list for a given term.

        Args:
            term (str): The term for which the posting list is requested.

        Returns:
            dict: The posting list for the term, or None if the term is not found.
        """
        if term in self.term_positions:
            offset = self.term_positions[term]
            block_filename = os.path.join(self.output_dir, 'inverted_index.bin')
            with open(block_filename, 'rb') as f:
                f.seek(offset)
                term_length = struct.unpack('I', f.read(4))[0]
                f.read(term_length)  # Skip term
                postings_length = struct.unpack('I', f.read(4))[0]
                postings = pickle.loads(f.read(postings_length))
                return postings
        else:
            return None
        # return self.inv_index_in_mem.get(term, None) # use this for test.py evaluation

    def save_index_data(self):
        """
        Save index data to disk.
        """
        self.save_mapping(self.term_positions, 'term_positions.bin')
        self.save_mapping(self.idf_values, 'idf.bin')
        self.save_mapping(dict(self.doc_lengths), 'doc_lengths.bin')
        self.save_mapping({'doc_count': self.doc_count}, 'doc_count.bin')
        self.save_mapping(self.doc_lengths_n, 'doc_lengths_n.bin')

    def load_index_data(self):
        """
        Load index data from disk.
        """
        self.term_positions = self.load_mapping('term_positions.bin')
        self.idf_values = self.load_mapping('idf.bin')
        self.doc_lengths = self.load_mapping('doc_lengths.bin')
        self.doc_count = self.load_mapping('doc_count.bin').get('doc_count', 0)
        self.doc_lengths_n = self.load_mapping('doc_lengths_n.bin')

    def save_mapping(self, dictionary: Dict[Any, Any], file_name: str):
        """
        Save mapping to disk.
        """
        try:
            _file = os.path.join(self.output_dir, file_name)
            with open(_file, 'wb') as f:
                pickle.dump(dictionary, f)
        except Exception as e:
            print(f"Error saving mapping to disk: {e}")

    def load_mapping(self, file_name: str):
        """
        Load term positions from disk.
        """
        try:
            _file = os.path.join(self.output_dir, file_name)
            if os.path.exists(_file):
                with open(_file, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"File {_file} does not exist.")
                return {}
        except Exception as e:
            print(f"Error loading mapping from disk: {e}")

    def compute_idf(self, num_occurrences):
        """
        Compute the Inverse Document Frequency (IDF) for a term.

        Args:
            num_occurrences (int): The number of documents in which the term occurs.

        Returns:
            float: The IDF value for the term.
        """
        return math.log(self.doc_count / num_occurrences, 10)

    def compute_tf(self, term_count):
        """
        Compute the Log Term Frequency for a term.

        Args:
            term_count (int): The number of times the term occurs in the document.

        Returns:
            float: The Log Term Frequency value for the term.
        """
        return 1 + math.log(term_count, 10)

    def compute_document_tfidf(self, document_as_list: List[str]) -> Dict[str, float]:
        """
        Compute the TF-IDF scores for a document. Special function for rocchio pipeline.

        Args:
            document_as_list (list): A list of terms in the document.

        Returns:
            dict: A dictionary of terms and their TF-IDF scores.
        """
        doc_tfidf: Dict[str, float] = {}

        for term in list(set(document_as_list)):
            # tfidf = tf * idf
            doc_tfidf[term] = (1+math.log(len([t for t in document_as_list if t == term]), 10)) * self.idf_values.get(
                term, 0.0)

        return doc_tfidf

    def remove_query_title_from_results(self, query_title: str, num_results: int, similar_documents_sorted: List[Tuple[int, float]]) -> List[Tuple[int, float] | None]:
        """
        Remove the query document from the list of most similar documents.

        Args:
            query_title (str): The title of the query document.
            num_results (int): The number of top documents to return.
            similar_documents_sorted (list): A list of tuples (doc_id, score) of the most similar documents.

        Returns:
            list: A list of tuples (doc_id, score) of the most similar documents, excluding the query document.
        """
        num_results_most_similar: List[Tuple[int, float]] = []
        idx = 0
        while len(num_results_most_similar) < num_results and idx < len(similar_documents_sorted):
            # if the document is not the query document, add it to the list of most similar documents
            if not self.document_titles[similar_documents_sorted[idx][0]] == query_title:
                num_results_most_similar.append(similar_documents_sorted[idx])
            idx += 1
        return num_results_most_similar

    def fast_cosine_score(self, query_terms: List[str] | Dict[str, float], k: int = 10, query_doc_title: str | None = None):
        """
        Compute the top K documents for a query using cosine similarity.

        Args:
            query_terms (List[str] | Dict[str, float]): A list of query terms or dictionary of query tf-idf values.
            k (int): The number of top documents to return.
            query_doc_title (str): The title of the query document.

        Returns:
            list: A list of tuples (doc_id, score) of the top K documents.
        """
        scores: dict[int, float] = defaultdict(float)

        query_tfidf: Dict[str, float] = {}
        if isinstance(query_terms, list):
            for term in list(set(query_terms)):
                # query_tfidf = tf * idf
                query_tfidf[term] = (1+math.log(len([t for t in query_terms if t == term]), 10)) * self.idf_values.get(
                    term, 0.0)
        elif isinstance(query_terms, dict):
            query_tfidf = query_terms
            query_terms = list(query_terms.keys())

        # normalize the query vector
        query_length = math.sqrt(sum([v**2 for v in query_tfidf.values()]))
        if query_length == 0:
            return []
        for term in query_tfidf:
            query_tfidf[term] /= query_length

        for term in query_terms:
            idf: float = self.idf_values.get(term, 0.0) # self.idf_values was calculated during indexing
            postings: Dict[int, float] = self.get_posting_list(term)

            if postings:
                for doc_id, tf in postings.items():
                    w_t_d = tf * idf # note: 'tf' here contains the value 1+log(tf_t,d)
                    w_t_q = query_tfidf[term]
                    scores[doc_id] += w_t_d * w_t_q # do add Wt,d * Wt,q to Scores[d]

        for doc_id in scores:
            scores[doc_id] /= self.doc_lengths_n[doc_id]

        # if the query document title is provided, this means the input query is the text associated with
        # a document by that title, and we need to remove that document from the results
        # (because it will always be the most similar document, since it's literally the same document).
        if query_doc_title:
            top_k_docs: List[Tuple[int, float]] = heapq.nlargest(k + 1, scores.items(), key=lambda item: item[1])
            top_k_docs = self.remove_query_title_from_results(query_title=query_doc_title, num_results=k, similar_documents_sorted=top_k_docs)
        else:
            top_k_docs: List[Tuple[int, float]] = heapq.nlargest(k, scores.items(), key=lambda item: item[1])

        return top_k_docs

    @staticmethod
    def rocchio_update(query_vector: Dict[str, float], relevant_docs: List[Dict[str, float]], non_relevant_docs: List[Dict[str, float]], alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.25) -> Dict[str, float]:
        """
        Update the query vector using the Rocchio algorithm.

        Args:
            query_vector (dict): The original query vector.
            relevant_docs (list): A list of relevant documents.
            non_relevant_docs (list): A list of non-relevant documents.
            alpha (float): Weight for the original query vector.
            beta (float): Weight for the relevant documents.
            gamma (float): Weight for the non-relevant documents.

        Returns:
            dict: The updated query vector.
        """
        updated_vector = defaultdict(float)

        # Add original query vector
        for term, weight in query_vector.items():
            updated_vector[term] += alpha * weight

        # Add relevant documents' vectors
        for doc_vector in relevant_docs:
            for term, weight in doc_vector.items():
                updated_vector[term] += (beta / len(relevant_docs)) * weight

        # Subtract non-relevant documents' vectors
        for doc_vector in non_relevant_docs:
            for term, weight in doc_vector.items():
                updated_vector[term] -= (gamma / len(non_relevant_docs)) * weight

        return updated_vector

    @staticmethod
    def mark_as_relevant(list_of_relevant_indices: List[int], top_k_docs: List[Tuple[int, float]]) -> Tuple[List[int], List[int]]:
        """
        Mark the relevant documents from the top K documents.

        Args:
            list_of_relevant_indices (list): A list of relevant document indices.
            top_k_docs (list): A list of tuples (doc_id, score) of the top K documents.

        Returns:
            tuple: A tuple containing the relevant and irrelevant document indices.
        """
        relevant_docs = []
        for idx in list_of_relevant_indices:
            relevant_docs.append(top_k_docs[idx][0])

        irrelevant_docs = [doc[0] for doc in top_k_docs if doc not in relevant_docs]
        return relevant_docs, irrelevant_docs

    def compute_tfidf_scores_for_doc_list(self, doc_list: List[int], documents: List[str]) -> List[Dict[str, float]]:
        """
        Compute the TF-IDF scores for a list of documents.

        Args:
            doc_list (list): A list of document indices.
            documents (list): A list of documents (strings).

        Returns:
            list: A list of dictionaries containing the TF-IDF scores for each document.
        """
        tfidf_scores: List[Dict[str, float]] = []
        for doc_id in doc_list:
            tfidf_scores.append(self.compute_document_tfidf(documents[doc_id].split()))

        return tfidf_scores

    def rocchio_pipeline(self, list_of_relevant_indices: List[int], top_docs: List[Tuple[int, float]], query: List[str], query_title: str, first_rocchio: bool = False) -> List[Tuple[int, float]]:
        """
        Perform the Rocchio algorithm pipeline to update the query vector and the top documents.

        Args:
            list_of_relevant_indices (list): A list of relevant document indices.
            top_docs (list): A list of tuples (doc_id, score) of the top K documents.
            query (list): The original query terms.
            query_title (str): The title of the query document.
            first_rocchio (bool): Indicates if this is the first Rocchio iteration.

        Returns:
            list: A list of tuples (doc_id, score) of the top documents after Rocchio.
        """

        # obtain doc indices (not cosine score)
        relevant_docs, irrelevant_docs = self.mark_as_relevant(list_of_relevant_indices=list_of_relevant_indices, top_k_docs=top_docs)

        tfidf_relevant_docs = self.compute_tfidf_scores_for_doc_list(doc_list=relevant_docs, documents=self.documents)
        tfidf_irrelevant_docs = self.compute_tfidf_scores_for_doc_list(doc_list=irrelevant_docs, documents=self.documents)

        if self.previous_rocchio_iteration_query is None or first_rocchio:
            # relevant docs are expected to be the dictionary for key: term, value: tfidf
            self.previous_rocchio_iteration_query = self.compute_document_tfidf(query)

        # apply rocchio
        rocchio_iteration_query: Dict[str, float] = self.rocchio_update(query_vector=self.previous_rocchio_iteration_query,
                                                             relevant_docs=tfidf_relevant_docs,
                                                             non_relevant_docs=tfidf_irrelevant_docs)

        self.previous_rocchio_iteration_query = rocchio_iteration_query

        # rocchio_iteration_query is now a vector and no longer a list of words (this was needed for rocchio)
        top_docs_after_rocchio = self.fast_cosine_score(query_terms=rocchio_iteration_query,
                                                        query_doc_title=query_title,
                                                        k=len(top_docs))

        return top_docs_after_rocchio

    @staticmethod
    def generate_token_stream(documents: List[str] | Generator[str, Any, None]) -> Generator[Tuple[str, int], Any, None]:
        """
        Generate a stream of (term, doc_id) tuples from a list of documents.

        Args:
            documents (list): A list of documents (strings).

        Yields:
            tuple: A tuple (term, doc_id) for each term in each document.
        """
        for doc_id, text in enumerate(documents):
            for term in text.split():
                yield term, doc_id

    @staticmethod
    def merge_dicts(c1: Dict[int, int], c2: Dict[int, int]) -> Dict[int, int]:
        """
        Merge two dictionaries by adding values of common keys.

        Args:
            c1 (dict): The first dictionary.
            c2 (dict): The second dictionary.

        Returns:
            dict: The merged dictionary.
        """

        c1 = {k: int(v) for k, v in c1.items()}
        c2 = {k: int(v) for k, v in c2.items()}

        # using default dict to simplify the addition logic
        merged_dict = defaultdict(int, c1)
        for k, v in c2.items():
            merged_dict[k] += v

        # convert back to a regular dict
        return dict(merged_dict)

def print_block_file(file_path: str) -> None:
    """
    Print the contents of a block file.

    Args:
        file_path (str): The path to the block file.
    """
    with open(file_path, 'rb') as f:
        while True:
            term_length_data = f.read(4)
            if not term_length_data:
                break
            term_length = struct.unpack('I', term_length_data)[0]
            term = f.read(term_length).decode('utf-8')
            postings_length = struct.unpack('I', f.read(4))[0]
            postings = pickle.loads(f.read(postings_length))

            print(f"Term: {term}")
            for doc_id, freq in postings.items():
                print(f"    Doc ID: {doc_id}, Frequency: {freq}")
