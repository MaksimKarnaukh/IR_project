import os
import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.index import DirectoryReader, IndexWriter, IndexWriterConfig
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import BooleanQuery


class PyLuceneWrapper:

    def __init__(self, recreated_index: bool = False, documents: dict = None ):

        # Initialize the JVM
        self.env = lucene.initVM()
        # Initialize the analyzer
        self.analyzer = StandardAnalyzer()
        # Initialize the index directory
        self.index_dir = None
        # set max clause count
        BooleanQuery.setMaxClauseCount(2048*10)

        if recreated_index or not os.path.exists(self.indexpath()):
            self.create_index(documents)
        else:
            self.index_dir = FSDirectory.open(Paths.get(self.indexpath()))

    def create_index(self, documents) -> None:
        """
        Create the index with the given documents.

        Args:
            documents (dict): A dictionary with the document titles as keys and the document texts as values
        """
        #     remove the index directory
        if os.path.exists(self.indexpath()):
            for file in os.listdir(self.indexpath()):
                os.remove(os.path.join(self.indexpath(), file))
            os.rmdir(self.indexpath())
        #     create the index directory
        os.makedirs(self.indexpath())
        self.index_dir = FSDirectory.open(Paths.get(self.indexpath()))

        # Initialize the index writer
        config = IndexWriterConfig(self.analyzer)
        writer = IndexWriter(self.index_dir, config)
        # Add documents to the index
        for title, text in documents.items():
            doc = Document()
            doc.add(TextField("title", title, Field.Store.YES))
            doc.add(TextField("content", text, Field.Store.YES))
            writer.addDocument(doc)
        writer.close()

    def indexpath(self):
        """
        Get the path to the index directory.

        Returns:
            str: The path to the index directory
        """
        index_path = "index"
        # if wd ends with src, remove src
        if os.getcwd().endswith("src"):
            index_path = "../" + index_path
        return index_path

    def search_index(self, query_string, num_results=10):
        """
        Search the index for the query string.

        Args:
            query_string (str): The query string
            num_results (int): The number of results to return

        Returns:
            list of tuples: The list of results, each result is a tuple with the title of the document and the score
        """
        searcher = IndexSearcher(DirectoryReader.open(self.index_dir))
        query_parser = QueryParser("content", self.analyzer)
        query = query_parser.parse(query_string)

        hits = searcher.search(query, num_results).scoreDocs
        results = []
        for hit in hits:
            doc_id = hit.doc
            doc = searcher.doc(doc_id)
            results.append((doc.get("title"), hit.score))
        return results
