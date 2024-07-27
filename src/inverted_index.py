import os
import heapq
import struct
from collections import defaultdict
import pickle


class SPIMI:
    def __init__(self, block_size_limit=100000, output_dir="./spimi_output_binary"):
        self.block_size_limit = block_size_limit
        self.output_dir = output_dir
        self.block_num = 0
        self.dictionary = defaultdict(set)
        self.token_count = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def write_block_to_disk(self):
        block_filename = os.path.join(self.output_dir, f"block_{self.block_num}.bin")
        with open(block_filename, 'wb') as f:
            for term, postings in self.dictionary.items():
                term_bytes = term.encode('utf-8')
                postings_bytes = pickle.dumps(postings)
                f.write(struct.pack('I', len(term_bytes)))
                f.write(term_bytes)
                f.write(struct.pack('I', len(postings_bytes)))
                f.write(postings_bytes)
        self.block_num += 1
        self.dictionary = defaultdict(set)
        self.token_count = 0
        return block_filename

    def read_next_term_postings(self, f):
        term_length_data = f.read(4)
        if not term_length_data:
            return None, None
        term_length = struct.unpack('I', term_length_data)[0]
        term = f.read(term_length).decode('utf-8')
        postings_length = struct.unpack('I', f.read(4))[0]
        postings_set = pickle.loads(f.read(postings_length))
        return term, postings_set

    def merge_blocks(self, block_files):
        merged_index = defaultdict(set)

        # Initialize a min-heap
        heap = []
        file_handles = []

        # Open all block files and push the first term from each block to the heap
        for block_num, block_file in enumerate(block_files):
            f = open(block_file, 'rb')
            file_handles.append(f)
            term, postings = self.read_next_term_postings(f)
            if term:
                heapq.heappush(heap, (term, block_num, postings))

        # Merge the blocks
        current_term = None
        current_postings = defaultdict(int)

        while heap:
            term, block_num, postings = heapq.heappop(heap)

            if term != current_term:
                if current_term is not None:
                    merged_index[current_term].update(current_postings.items())
                current_term = term
                current_postings = defaultdict(int, postings)
            else:
                for doc_id, freq in postings:
                    current_postings[doc_id] += freq

            term, postings = self.read_next_term_postings(file_handles[block_num])
            if term:
                heapq.heappush(heap, (term, block_num, postings))

        if current_term is not None:
            merged_index[current_term].update(current_postings.items())

        for f in file_handles:
            f.close()

        final_index_filename = os.path.join(self.output_dir, "final_index.bin")
        with open(final_index_filename, 'wb') as f:
            for term, postings in merged_index.items():
                term_bytes = term.encode('utf-8')
                postings_bytes = pickle.dumps(postings)
                f.write(struct.pack('I', len(term_bytes)))
                f.write(term_bytes)
                f.write(struct.pack('I', len(postings_bytes)))
                f.write(postings_bytes)

        return final_index_filename

    def spimi_invert(self, token_stream):
        block_files = []

        for term, doc_id in token_stream:
            if self.token_count >= self.block_size_limit:
                block_files.append(self.write_block_to_disk())

            # Update the term frequency in the postings list
            found = False
            for pair in self.dictionary[term]:
                if pair[0] == doc_id:
                    self.dictionary[term].remove(pair)
                    self.dictionary[term].add((doc_id, pair[1] + 1))
                    found = True
                    break
            if not found:
                self.dictionary[term].add((doc_id, 1))

            self.token_count += 1

        if self.token_count > 0:
            block_files.append(self.write_block_to_disk())

        final_index_filename = self.merge_blocks(block_files)

        return final_index_filename

def generate_token_stream(documents):
    for doc_id, text in enumerate(documents):
        for term in text.split():
            yield term, doc_id

def merge_dicts():
    from collections import defaultdict

    c1 = {'A': 4, 'T': 6}
    c2 = {'L': 2, 'T': 1}

    c1 = {k: int(v) for k, v in c1.items()}
    c2 = {k: int(v) for k, v in c2.items()}

    # Use defaultdict to simplify the addition logic
    merged_dict = defaultdict(int, c1)
    for k, v in c2.items():
        merged_dict[k] += v

    # Convert defaultdict back to a regular dict
    merged_dict = dict(merged_dict)

    print(merged_dict)


# Example usage
# if __name__ == "__main__":
#     documents = [
#         "the cat in the hat",
#         "the quick brown fox",
#         "the lazy dog",
#         "the hat is in the cat",
#     ]
#
#     token_stream = generate_token_stream(documents)
#     spimi = SPIMI(block_size_limit=10)
#     final_index_filename = spimi.spimi_invert(token_stream)
#     print(f"Final index written to: {final_index_filename}")

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
    spimi = SPIMI(block_size_limit=10)
    final_index_filename = spimi.spimi_invert(token_stream)
    print(f"Final index written to: {final_index_filename}")
