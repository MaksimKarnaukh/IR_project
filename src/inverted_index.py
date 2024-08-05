import os
import heapq
import struct
from collections import defaultdict
import pickle
import math
from src import variables

DEFAULT_BLOCK_SIZE_LIMIT = 10000

class SPIMI:
    def __init__(self, block_size_limit=DEFAULT_BLOCK_SIZE_LIMIT, output_dir=variables.inverted_index_folder):
        self.block_size_limit = block_size_limit
        self.output_dir = output_dir
        self.block_num = 0
        self.dictionary = dict() # {term: {doc_id: freq}}
        self.token_count = 0
        self.term_positions = {}  # {term: offset}

        self.doc_lengths = defaultdict(int)  # {doc_id: doc_length}
        self.doc_count = 0  # total number of documents

        os.makedirs(self.output_dir, exist_ok=True)

    # function for writing the term and postings to disk
    def write_index_entry(self, term, postings, f):
        try:
            term_bytes = term.encode('utf-8')
            postings_bytes = pickle.dumps(dict(postings))
            f.write(struct.pack('I', len(term_bytes)))
            f.write(term_bytes)
            f.write(struct.pack('I', len(postings_bytes)))
            f.write(postings_bytes)
        except Exception as e:
            print(f"Error writing term '{term}' to disk: {e}")

    def write_block_to_disk(self):
        block_filename = os.path.join(self.output_dir, f"block_{self.block_num}.bin")
        with open(block_filename, 'wb') as f:
            sorted_terms = sorted(self.dictionary.items())
            for term, postings in sorted_terms:
                self.write_index_entry(term, postings, f)

        self.block_num += 1
        self.dictionary = dict()
        self.token_count = 0
        return block_filename

    def read_next_term_postings(self, f):
        term_length_data = f.read(4)
        if not term_length_data:
            return None, None
        term_length = struct.unpack('I', term_length_data)[0]
        term = f.read(term_length).decode('utf-8')
        postings_length = struct.unpack('I', f.read(4))[0]
        postings = pickle.loads(f.read(postings_length))
        return term, postings

    def merge_two_blocks(self, block_file1, block_file2, is_last_merge=False):
        merged_index = defaultdict(dict)

        f1 = open(block_file1, 'rb')
        f2 = open(block_file2, 'rb')

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
                    offset = f.tell()

        self.block_num += 1
        return merged_block_filename

    def merge_blocks(self, block_files, delete_merged_blocks=True):
        while len(block_files) > 1:
            new_block_files = []

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

        final_block = block_files[0]
        final_index_filename = os.path.join(self.output_dir, 'inverted_index.bin')
        os.rename(final_block, final_index_filename)
        return final_index_filename

    def spimi_invert(self, token_stream):
        block_files = []

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
            self.doc_lengths[doc_id] += 1 ###

        if self.token_count > 0:
            block_files.append(self.write_block_to_disk())

        final_index_filename = self.merge_blocks(block_files)

        return final_index_filename

    def get_posting_list(self, term):
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

    def save_term_positions(self):
        try:
            term_positions_file = os.path.join(self.output_dir, 'term_positions.bin')
            with open(term_positions_file, 'wb') as f:
                pickle.dump(self.term_positions, f)
        except Exception as e:
            print(f"Error saving term positions to disk: {e}")

    def load_term_positions(self):
        try:
            term_positions_file = os.path.join(self.output_dir, 'term_positions.bin')
            if os.path.exists(term_positions_file):
                with open(term_positions_file, 'rb') as f:
                    self.term_positions = pickle.load(f)
            else:
                self.term_positions = {}
        except Exception as e:
            print(f"Error loading term positions from disk: {e}")

    def compute_idf(self, term):
        postings = self.get_posting_list(term)
        if postings:
            doc_freq = len(postings)
            return math.log(self.doc_count / doc_freq)
        else:
            return 0.0

    def compute_tf(self, term, doc_id):
        postings = self.get_posting_list(term)
        if postings and doc_id in postings:
            return 1 + math.log(postings[doc_id])
        else:
            return 0.0

    def fast_cosine_score(self, query_terms, K=10):
        Scores = defaultdict(float)

        for term in query_terms:
            idf = self.compute_idf(term)
            postings = self.get_posting_list(term)

            if postings:
                for doc_id, tf in postings.items():
                    Scores[doc_id] += self.compute_tf(term, doc_id) * idf

        for doc_id in Scores:
            Scores[doc_id] /= self.doc_lengths[doc_id] # /= math.sqrt(self.doc_lengths[doc_id])

        top_K_docs = heapq.nlargest(K, Scores.items(), key=lambda item: item[1])

        return top_K_docs


def generate_token_stream(documents):
    for doc_id, text in enumerate(documents):
        for term in text.split():
            yield term, doc_id

def merge_dicts(c1, c2):
    from collections import defaultdict

    c1 = {k: int(v) for k, v in c1.items()}
    c2 = {k: int(v) for k, v in c2.items()}

    # Use defaultdict to simplify the addition logic
    merged_dict = defaultdict(int, c1)
    for k, v in c2.items():
        merged_dict[k] += v

    # Convert defaultdict back to a regular dict
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
    token_stream = generate_token_stream(documents)
    spimi = SPIMI(block_size_limit=3)
    final_index_filename = spimi.spimi_invert(token_stream)
    print(f"Final index written to: {final_index_filename}")

    # print contents of all binary files in the output/spimi_output directory
    for file in os.listdir('../output/spimi_output'):
        if file.endswith(".bin"):
            print("-----\n")
            print("\nfile:", file)
            print_block_file(os.path.join('../output/spimi_output', file))
            print("-----\n")

    # Print contents of the final index file
    print("\nFinal index contents:")
    print_block_file(final_index_filename)

    # Test retrieving the posting list for a term
    term = "then"
    postings = spimi.get_posting_list(term)
    print(f"\nPosting list for term '{term}': {postings}")

    # Test saving and loading term positions
    spimi.save_term_positions()
    spimi.load_term_positions()
    term = "cat"
    postings = spimi.get_posting_list(term)
    print(f"\nPosting list for term '{term}': {postings}")

    # Perform ranked retrieval with cosine similarity
    query = ["then", "cat"]
    top_docs = spimi.fast_cosine_score(query, K=4)
    print(f"\nTop documents for query '{query}': {top_docs}")