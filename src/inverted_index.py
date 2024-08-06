import os
import heapq
import struct
from collections import defaultdict
import pickle
import math
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
        dictionary (Dict[str, Dict[int, int]]): Dictionary of terms and their postings.
        token_count (int): Number of tokens in the current block.
        term_positions (Dict[str, int]): mapping of term positions in the index file.
        doc_lengths (Dict[int, int]): Dictionary of document lengths.
        doc_count (int): Total number of documents.
        idf_values (Dict[str, float]): Dictionary of IDF values for terms.

    Args:
        block_size_limit (int): Maximum number of tokens in a block before writing to disk.
        output_dir (str): Directory where the index file(s) will be stored.

    methods:
        write_index_entry(self, term: str, postings: Dict[int, int], f: BinaryIO)
        write_block_to_disk(self)
        read_next_term_postings(self, f)
        merge_two_blocks(self, block_file1, block_file2, is_last_merge=False)
        merge_blocks(self, block_files, delete_merged_blocks=True)
        spimi_invert(self, token_stream)
        get_posting_list(self, term)
        save_term_positions(self)
        load_term_positions(self)
        compute_idf(self, term)
        compute_tf(self, term, doc_id)
        fast_cosine_score(self, query_terms, K=10)
    """

    def __init__(self, block_size_limit: int = DEFAULT_BLOCK_SIZE_LIMIT, output_dir: str = variables.inverted_index_folder):
        """
        Initialize the SPIMI (Single-Pass In-Memory Indexing) instance.

        Args:
            block_size_limit (int): Maximum number of tokens in a block before writing to disk.
            output_dir (str): Directory where the index file(s) will be stored.
        """
        self.block_size_limit: int = block_size_limit
        self.output_dir: str = output_dir
        self.block_num: int = 0
        self.dictionary: Dict[str, Dict[int, int]] = dict() # {term: {doc_id: freq}}
        self.token_count: int = 0
        self.term_positions: Dict[str, int] = {}  # {term: offset}

        self.doc_lengths: Dict[int, int] = defaultdict(int)  # {doc_id: doc_length}
        self.doc_count: int = 0  # total number of documents

        self.idf_values: Dict[str, float] = {}  # {term: idf_value}

        os.makedirs(self.output_dir, exist_ok=True)

    def write_index_entry(self, term: str, postings: Dict[int, int], f: BinaryIO):
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

    def write_block_to_disk(self) -> str:
        """
        Write the current in-memory term dictionary to a block file on disk.

        Returns:
            str: The filename of the block file.
        """
        block_filename = os.path.join(self.output_dir, f"block_{self.block_num}.bin")
        with open(block_filename, 'wb') as f:
            sorted_terms = sorted(self.dictionary.items())
            for term, postings in sorted_terms:
                self.write_index_entry(term, postings, f)

        self.block_num += 1
        self.dictionary = dict()
        self.token_count = 0
        return block_filename

    def read_next_term_postings(self, f: BinaryIO) -> Tuple[str | None, Dict[int, int] | None]:
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
        term_length = struct.unpack('I', term_length_data)[0]
        term = f.read(term_length).decode('utf-8')
        postings_length = struct.unpack('I', f.read(4))[0]
        postings = pickle.loads(f.read(postings_length))
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
        merged_index: Dict[str, Dict[int, int]] = defaultdict(dict)

        f1: BinaryIO = open(block_file1, 'rb')
        f2: BinaryIO = open(block_file2, 'rb')

        term1, postings1 = self.read_next_term_postings(f1)
        term2, postings2 = self.read_next_term_postings(f2)

        while term1 is not None or term2 is not None:
            if term1 is not None and (term2 is None or term1 < term2):
                merged_index[term1] = postings1
                term1, postings1 = self.read_next_term_postings(f1)
            elif term2 is not None and (term1 is None or term2 < term1):
                merged_index[term2] = postings2
                term2, postings2 = self.read_next_term_postings(f2)
            else:
                merged_index[term1] = merge_dicts(postings1, postings2)
                term1, postings1 = self.read_next_term_postings(f1)
                term2, postings2 = self.read_next_term_postings(f2)

        f1.close()
        f2.close()

        merged_block_filename = os.path.join(self.output_dir, f"merged_{self.block_num}.bin")
        with open(merged_block_filename, 'wb') as f:
            offset = f.tell()
            for term, postings in merged_index.items():
                self.write_index_entry(term, postings, f)
                if is_last_merge:
                    self.term_positions[term] = offset
                    self.idf_values[term] = self.compute_idf(postings)
                offset = f.tell()

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
            new_block_files: List[str] = []

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

        self.save_mapping(self.idf_values, 'idf.bin')

        return final_index_filename

    # def compute_all_idf(self):
    #     """
    #     Compute the IDF values for all terms in the final merged index.
    #     """
    #     for term, offset in self.term_positions.items():
    #         postings = self.get_posting_list(term)
    #         if postings:
    #             self.idf_values[term] = math.log(self.doc_count / len(postings))
    #     self.save_mapping(self.idf_values, 'idf.bin')

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
            self.doc_count = max(self.doc_count, doc_id + 1)
            if term not in self.dictionary:

                if self.token_count + 1 > self.block_size_limit:
                    block_files.append(self.write_block_to_disk())

                self.dictionary[term] = dict()
                self.token_count += 1

            if doc_id not in self.dictionary[term]:
                self.dictionary[term][doc_id] = 0

            self.dictionary[term][doc_id] += 1
            self.doc_lengths[doc_id] += 1

        if self.token_count > 0:
            block_files.append(self.write_block_to_disk())

        final_index_filename: str = self.merge_blocks(block_files)

        return final_index_filename

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

    def save_mapping(self, dictionary: Dict[str, int | float], file_name: str):
        """
        Save mapping to disk.
        """
        try:
            term_positions_file = os.path.join(self.output_dir, file_name)
            with open(term_positions_file, 'wb') as f:
                pickle.dump(dictionary, f)
        except Exception as e:
            print(f"Error saving term positions to disk: {e}")

    def load_mapping(self, file_name: str):
        """
        Load term positions from disk.
        """
        try:
            term_positions_file = os.path.join(self.output_dir, file_name)
            if os.path.exists(term_positions_file):
                with open(term_positions_file, 'rb') as f:
                    return pickle.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error loading term positions from disk: {e}")

    def compute_idf(self, postings):
        if postings:
            return math.log(self.doc_count / len(postings))
        else:
            return 0.0

    def compute_tf(self, term, doc_id):
        postings = self.get_posting_list(term)
        if postings and doc_id in postings:
            return 1 + math.log(postings[doc_id])
        else:
            return 0.0

    def fast_cosine_score(self, query_terms, K=10):
        """
        Compute the top K documents for a query using cosine similarity.

        Args:
            query_terms (list): A list of query terms.
            K (int): The number of top documents to return.

        Returns:
            list: A list of tuples (doc_id, score) of the top K documents.
        """
        Scores = defaultdict(float)

        for term in query_terms:
            idf = self.idf_values.get(term, 0.0)
            postings = self.get_posting_list(term)

            if postings:
                for doc_id, tf in postings.items():
                    Scores[doc_id] += self.compute_tf(term, doc_id) * idf

        for doc_id in Scores:
            Scores[doc_id] /= self.doc_lengths[doc_id] # /= math.sqrt(self.doc_lengths[doc_id])

        top_K_docs = heapq.nlargest(K, Scores.items(), key=lambda item: item[1])

        return top_K_docs


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

def print_block_file(file_path):
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


# Example usage
if __name__ == "__main__":
    documents = [
        "then cat inn then hat",
        "then quick brown fox",
        "then lazy dog",
        "then hat sis inn then cat",
    ]
    # the cat in the hat quick brown fox lazy dog
    """
    BLOCK 1:
    then: d1
    cat: d1
    inn: d1
    
    FLUSH
    
    BLOCK 2:
    hat: d1
    then: d2
    quick: d2
    
    FLUSH
    
    BLOCK 3:
    brown: d2
    fox: d2
    then: d3
    
    FLUSH
    
    BLOCK 4:
    lazy: d3
    dog: d3
    then: d4
    
    FLUSH
    
    BLOCK 5:
    hat: d4
    sis: d4
    inn: d4
    
    FLUSH
    
    BLOCK 6:
    then: d4
    cat: d4
    
    MERGE BLOCK1 and BLOCK2 into BLOCK12
    
    then: d1, d2
    cat: d1
    inn: d1
    hat: d1
    quick: d2
    
    MERGE BLOCK3 and BLOCK4 into BLOCK34
    
    brown: d2
    fox: d2
    then: d3, d4
    lazy: d3
    dog: d3
    
    MERGE BLOCK5 and BLOCK6 into BLOCK56
    
    hat: d4
    sis: d4
    inn: d4
    then: d4
    cat: d4
    
    MERGE BLOCK12 and BLOCK34 into BLOCK1234
    
    then: d1, d2, d3, d4
    cat: d1
    inn: d1
    hat: d1
    quick: d2
    brown: d2
    fox: d2
    lazy: d3
    dog: d3
    
    MERGE BLOCK1234 and BLOCK56 into BLOCK123456
    
    then: d1, d2, d3, d4
    cat: d1, d4
    inn: d1, d4
    hat: d1, d4
    quick: d2
    brown: d2
    fox: d2
    lazy: d3
    dog: d3
    sis: d4
    
    
    """

    # token_stream = generate_token_stream(documents)
    # final_index_filename = spimi_invert(token_stream, output_dir="./spimi_output_binary", block_size_limit=10)
    # print(f"Final index written to: {final_index_filename}")
    token_stream_ = generate_token_stream(documents)
    spimi = SPIMI(block_size_limit=3)
    final_index_filename_ = spimi.spimi_invert(token_stream_)
    print(f"Final index written to: {final_index_filename_}")

    # print contents of all binary files in the output/spimi_output directory
    for file in os.listdir('../output/spimi_output'):
        if file.endswith("index.bin"):
            print("-----\n")
            print("\nfile:", file)
            print_block_file(os.path.join('../output/spimi_output', file))
            print("-----\n")

    # Print contents of the final index file
    print("\nFinal index contents:")
    print_block_file(final_index_filename_)

    # Test retrieving the posting list for a term
    term_ = "then"
    postings_ = spimi.get_posting_list(term_)
    print(f"\nPosting list for term '{term_}': {postings_}")

    # Test saving and loading term positions
    spimi.save_mapping(spimi.term_positions, 'term_positions.bin')
    spimi.term_positions = spimi.load_mapping('term_positions.bin')
    term_ = "cat"
    postings_ = spimi.get_posting_list(term_)
    print(f"\nPosting list for term '{term_}': {postings_}")

    # Perform ranked retrieval with cosine similarity
    query_ = ["then", "cat"]
    top_docs = spimi.fast_cosine_score(query_, K=4)
    print(f"\nTop documents for query '{query_}': {top_docs}")