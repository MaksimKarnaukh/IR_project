import tkinter as tk
from tkinter import ttk
from test import calculate_precision, calculate_recall
from utils import *
from src import variables
from pylucene import PyLuceneWrapper
from data_preprocessor import DataPreprocessor
from typing import List, Tuple, Dict, Any

class SimpleGUI:
    """
    Simple GUI for the similar documents retrieval system.

    Attributes:
        doc_dict (Dict[str, str]): Dictionary of document titles and texts.
        documents (List[str]): List of document texts.
        document_titles (List[str]): List of document titles.
        retrieval_system: The retrieval system to use.
        lucene_retrieval_system: The PyLucene retrieval system.
        preprocessor: The data preprocessor.
        ...
        current_query: The current query.
        current_by_title: Whether the current query is a title or not.
        current_similar_documents: The current similar documents.
        first_rocchio: The flag for the first Rocchio feedback.
    """

    def return_similar_documents(self, doc_dict: Dict[str, str], query: str, by_title: bool = False, num_results: int = 10) -> Tuple[List[str], List[str], float, float] | Tuple[List[str], List[str], float, float, float, float]:
        """
        Return similar documents to the query (and the precision and recall if applicable).

        Args:
            doc_dict (Dict[str, str]): Dictionary of document titles and texts.
            query (str): Query to find similar documents for.
            by_title (bool): Whether the query is a title or not.
            num_results (int): Number of results to return.

        Returns:
            List of similar document titles, List of similar documents, Precision, Recall
        """

        similar_documents_titles, similar_documents = [], []
        if by_title:
            query_document = doc_dict[query]
            query_document_split = query_document.split()
            similar_documents = self.retrieval_system.fast_cosine_score(query_document_split, k=num_results, query_doc_title=query)
            similar_documents_titles = [self.retrieval_system.document_titles[tup[0]] for tup in similar_documents]
        else:
            query_document = self.preprocessor.preprocess_text(query)
            if isinstance(query, str):
                query = self.preprocessor.preprocess_text(query).split()
            similar_documents = self.retrieval_system.fast_cosine_score(query, k=num_results, query_doc_title=None)
            similar_documents_titles = [self.retrieval_system.document_titles[tup[0]] for tup in similar_documents]

        expected_lucene = [tup[0] for tup in self.lucene_retrieval_system.search_index(query_document, num_results=num_results*2)]
        par_lucene = calculate_precision(expected_lucene, similar_documents_titles), calculate_recall(expected_lucene, similar_documents_titles)

        print(f"Expected Lucene: {expected_lucene}")
        print(f"Retrieved: {similar_documents_titles}")

        if by_title:
            title = query
            d = read_gt(variables.filepath_path_gt)  # Read ground truth

            if title not in d.keys():
                return similar_documents_titles, similar_documents, par_lucene[0], par_lucene[1]

            expected_gt = list(d[title].keys())
            par_gt = calculate_precision(expected_gt, similar_documents_titles), calculate_recall(expected_gt, similar_documents_titles)

            print(f"Expected GT: {expected_gt}")

            return similar_documents_titles, similar_documents, par_gt[0], par_gt[1], par_lucene[0], par_lucene[1]

        return similar_documents_titles, similar_documents, par_lucene[0], par_lucene[1]

    def center_window(self, window_width=1450, window_height=850) -> None:
        """
        Center the window on the screen.

        Args:
            window_width (int): Width of the window.
            window_height (int): Height of the window.
        """
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        # calculate position x and y coordinates
        x = (screen_width / 2) - (window_width / 2)
        y = (screen_height / 2) - (window_height / 2)
        self.root.geometry('%dx%d+%d+%d' % (window_width, window_height, x, y))

    def process_input(self) -> None:
        """
        Process the input query and display the similar documents.
        """

        def process_input_option(input_text: str, by_title: bool) -> None:
            """
            Process the input text and display the similar documents.

            Args:
                input_text (str): Input text to process.
                by_title (bool): Whether the input text is a title or not.
            """

            # check if self.num_results_entry.get() is a number in string format
            num_results = 10
            if self.num_results_entry.get().isdigit():
                num_results = int(self.num_results_entry.get())
            result = self.return_similar_documents(self.doc_dict, input_text, by_title=by_title, num_results=num_results)

            output_text2 = ""
            self.clear_result_checkboxes()

            if len(result) == 6:
                similar_documents_titles, similar_documents, par0_gt, par1_gt, par0_lucene, par1_lucene = result

                output_text1 = ""
                output_text2 += f"Results vs Ground Truths:\n" \
                               f"Precision: {par0_gt}\n" \
                               f"Recall: {par1_gt}\n"

                for i, doc in enumerate(similar_documents_titles, 1):
                    output_text1 += f"{i}. {doc}\n"
                    self.add_result_checkbox(doc, i)

            else:
                similar_documents_titles, similar_documents, par0_lucene, par1_lucene = result
                output_text1 = f""
                for i, doc in enumerate(similar_documents_titles, 1):
                    output_text1 += f"{i}. {doc}\n"
                    self.add_result_checkbox(doc, i)

            output_text2 += f"Results vs Lucene:\n" \
                            f"Precision: {par0_lucene}\n" \
                            f"Recall: {par1_lucene}\n"

            self.output_box1.insert(tk.END, output_text1 + "\n")
            self.output_box2.insert(tk.END, output_text2 + "\n")

            self.current_query = input_text
            self.current_by_title = by_title
            self.current_similar_documents = similar_documents

        selected_option = self.choice_var.get()

        self.output_box1.delete(1.0, tk.END)
        self.output_box2.delete(1.0, tk.END)

        self.first_rocchio = True # Reset the first Rocchio flag if we are processing a new query

        if selected_option == "Query Sentence":
            input_ = self.entry.get()
            process_input_option(input_, by_title=False)

        elif selected_option == "Title":
            selected_title = self.title_combobox.get()
            process_input_option(selected_title, by_title=True)

    def add_result_checkbox(self, doc_title, index) -> None:
        """
        Add a checkbox for each result document.

        Args:
            doc_title (str): Document title to add to the checkbox.
            index (int): The index of the document to position the checkbox in the grid.
        """
        var = tk.IntVar()
        column = (index - 1) % 5  # 5 columns
        row = (index - 1) // 5
        checkbox = tk.Checkbutton(self.result_frame, text=doc_title, variable=var, font=("Helvetica", 12))
        checkbox.var = var
        checkbox.grid(row=row, column=column, sticky=tk.W, padx=5, pady=5)
        self.result_checkboxes.append(checkbox)

    def clear_result_checkboxes(self) -> None:
        """
        Clear all the checkboxes in the result frame.
        """
        for checkbox in self.result_checkboxes:
            checkbox.grid_forget()
        self.result_checkboxes.clear()

    def mark_as_relevant(self) -> None:
        """
        Function to handle the "Mark as Relevant" button click. After selecting the relevant documents,
        the Rocchio relevance feedback is applied and the results are updated.
        """

        # Get indices of the selected relevant documents relative to the current displayed list
        relevant_indices = [index for index, checkbox in enumerate(self.result_checkboxes) if checkbox.var.get() == 1]

        # if by title, we need to equal query to the document text of the selected title
        if self.current_by_title:
            query = self.doc_dict[self.current_query].split()
        else:
            query = self.preprocessor.preprocess_text(self.current_query).split()

        # Perform the Rocchio relevance feedback
        updated_top_docs = self.retrieval_system.rocchio_pipeline(list_of_relevant_indices=relevant_indices,
                                                                  top_docs=self.current_similar_documents,
                                                                  query=query,
                                                                  query_title=self.current_query if self.current_by_title else None,
                                                                  first_rocchio=self.first_rocchio)

        self.first_rocchio = False

        # Update the displayed results after Rocchio adjustment
        self.update_results_after_rocchio(updated_top_docs)

    def update_results_after_rocchio(self, top_docs) -> None:
        """
        Update the result list after applying the Rocchio pipeline.

        Args:
            top_docs (List[Tuple[int, float]]): List of top documents after Rocchio feedback.
        """
        self.clear_result_checkboxes()

        updated_doc_titles = [self.document_titles[tup[0]] for tup in top_docs]

        output_text = ""
        for i, doc in enumerate(updated_doc_titles, 1):
            output_text += f"{i}. {doc}\n"
            self.add_result_checkbox(doc, i)

        self.output_box1.delete(1.0, tk.END)
        self.output_box2.delete(1.0, tk.END)

        self.output_box1.insert(tk.END, output_text + "\n")
        self.output_box2.insert(tk.END, "Results updated with Rocchio feedback.\n")

    def __init__(self, root, doc_dict, retrieval_system):
        """
        Initialize the GUI.

        Args:
            root: The root window.
            doc_dict: Dictionary of document titles and texts.
            retrieval_system: The retrieval system to use.
        """

        self.doc_dict = doc_dict
        self.documents = list(doc_dict.values())
        self.document_titles = list(doc_dict.keys())

        self.retrieval_system = retrieval_system # our retrieval system using SPIMI
        self.lucene_retrieval_system = PyLuceneWrapper(documents=doc_dict) # PyLucene retrieval system
        self.preprocessor = DataPreprocessor()

        self.root = root
        root.title("Similar Documents Retrieval System")

        # four variables to keep track of the current state of the GUI for the Rocchio relevance feedback pipeline
        self.current_query = None
        self.current_by_title = None
        self.current_similar_documents = []
        self.first_rocchio = True

        # Center the window
        self.center_window()

        # Set a custom style for the buttons
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 14))

        # Dropdown for choosing either Query Sentence or Title
        self.choice_var = tk.StringVar()
        self.choice_var.set("Query Sentence")
        self.choice_label = ttk.Label(root, text="Choose an input option:", font=("Helvetica", 16))
        self.choice_label.pack()
        self.choice_combobox = ttk.Combobox(root, values=["Query Sentence", "Title"], textvariable=self.choice_var)
        self.choice_combobox.pack(pady=10)

        # Input Entry
        self.entry_label = ttk.Label(root, text="Query input:", font=("Helvetica", 16))
        self.entry_label.pack()
        self.entry = ttk.Entry(root, font=("Helvetica", 14))
        self.entry.pack(pady=10)

        # A list of titles
        titles = sorted(self.document_titles)
        # Create a title Combobox
        self.title_combobox = ttk.Combobox(root, values=titles, font=("Helvetica", 14), state="readonly")
        self.title_combobox.pack(pady=10)

        # Additional Input for Number of Results
        self.num_results_label = ttk.Label(root, text="Number of Results:", font=("Helvetica", 16))
        self.num_results_label.pack()
        self.num_results_entry = ttk.Entry(root, font=("Helvetica", 14))
        self.num_results_entry.pack(pady=10)

        # Process Input Button
        self.process_button = ttk.Button(root, text="Process Input", command=self.process_input, style="TButton")
        self.process_button.pack(pady=15)

        # Result Checkboxes Frame
        self.result_frame = tk.Frame(root)
        self.result_frame.pack(pady=10)
        self.result_checkboxes = []

        # "Mark as Relevant" Button
        self.mark_relevant_button = ttk.Button(root, text="Mark as Relevant", command=self.mark_as_relevant, style="TButton")
        self.mark_relevant_button.pack(pady=15)

        # Output Text Box
        self.output_label1 = ttk.Label(root, text="Similar documents:", font=("Helvetica", 16))
        self.output_label1.pack()
        self.output_box1 = tk.Text(root, height=10, width=60, wrap=tk.WORD, font=("Helvetica", 12))
        self.output_box1.pack(padx=10, pady=5)

        # Output Text Box
        self.output_label2 = ttk.Label(root, text="Performance statistics:", font=("Helvetica", 16))
        self.output_label2.pack()
        self.output_box2 = tk.Text(root, height=8, width=60, wrap=tk.WORD, font=("Helvetica", 12))
        self.output_box2.pack(padx=10, pady=5)


def run_gui(doc_dict, retrieval_system):
    root = tk.Tk()
    app = SimpleGUI(root, doc_dict, retrieval_system)
    root.mainloop()


