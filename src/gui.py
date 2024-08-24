import tkinter as tk
from tkinter import ttk

from test import calculatePrecisionAndRecall
from utils import *
from related_doc_retrieval import *
from src import variables


class SimpleGUI:

    def return_similar_documents(self, doc_dict, query: str, by_title: bool = False, num_results: int = 10):
        """
        Return similar documents to the query (and the precision and recall if applicable).
        :param doc_dict: dictionary of documents
        :param query: query
        :param by_title: if True, the query is a title. Otherwise, the query is a text.
        :param num_results: number of results to return
        :return:
        """

        similar_documents_titles, similar_documents = [], []
        if by_title:
            query_document = doc_dict[query]
            if isinstance(query, str):
                query_document = query_document.split()
            similar_documents = self.retrieval_system.fast_cosine_score(query_document, k=10, query_doc_title=query)
            similar_documents_titles = [self.retrieval_system.document_titles[tup[0]] for tup in similar_documents]
            similar_documents = [tup[0] for tup in similar_documents]
            print(similar_documents_titles, similar_documents)
        else:
            if isinstance(query, str):
                query = query.split()
            similar_documents = self.retrieval_system.fast_cosine_score(query, k=10, query_doc_title=None)
            similar_documents_titles = [self.retrieval_system.document_titles[tup[0]] for tup in similar_documents]
            similar_documents = [tup[0] for tup in similar_documents]

        if by_title:
            title = query
            d = read_gt(variables.filepath_path_gt)  # Read ground truth

            if title not in d.keys():
                return similar_documents_titles, similar_documents

            expected = list(d[title].keys())
            par = calculatePrecisionAndRecall(expected, similar_documents_titles)

            return similar_documents_titles, similar_documents, par[0], par[1]

        return similar_documents_titles, similar_documents

    def center_window(self, window_width=650, window_height=700):
        """
        Center the window on the screen.
        :param window_width: window width
        :param window_height: window height
        :return:
        """
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        # calculate position x and y coordinates
        x = (screen_width/2) - (window_width/2)
        y = (screen_height/2) - (window_height/2)
        self.root.geometry('%dx%d+%d+%d' % (window_width, window_height, x, y))

    def process_input(self):

        def process_input_option(input_text, by_title: bool):

            # check if self.num_results_entry.get() is a number in string format
            num_results = 10
            if self.num_results_entry.get().isdigit():
                num_results = int(self.num_results_entry.get())
            result = self.return_similar_documents(self.doc_dict, input_text, by_title=by_title, num_results=num_results)
            if len(result) == 4:
                similar_documents_titles, similar_documents, par0, par1 = result

                output_text1 = ""
                output_text2 = f"Precision: {par0}\n" \
                               f"Recall: {par1}\n"

                for i, doc in enumerate(similar_documents_titles, 1):
                    output_text1 += f"{i}. {doc}\n"
                self.output_box1.insert(tk.END, output_text1 + "\n")
                self.output_box2.insert(tk.END, output_text2 + "\n")
            else:
                similar_documents_titles, similar_documents = result
                output_text1 = f""
                output_text2 = f"Title not found in ground truth."
                for i, doc in enumerate(similar_documents_titles, 1):
                    output_text1 += f"{i}. {doc}\n"
                self.output_box1.insert(tk.END, output_text1 + "\n")
                self.output_box2.insert(tk.END, output_text2 + "\n")

        selected_option = self.choice_var.get()

        self.output_box1.delete(1.0, tk.END)
        self.output_box2.delete(1.0, tk.END)

        if selected_option == "Query Sentence":
            input_text = self.entry.get()
            process_input_option(input_text, by_title=False)
            # self.entry.delete(0, tk.END) # uncomment this line to clear the input box after processing

        elif selected_option == "Title":
            selected_title = self.title_combobox.get()
            process_input_option(selected_title, by_title=True)

    def __init__(self, root, doc_dict, retrieval_system):

        self.doc_dict = doc_dict
        self.documents = list(doc_dict.values())
        self.document_titles = list(doc_dict.keys())

        self.retrieval_system = retrieval_system

        self.root = root
        root.title("Similar Documents Retrieval System")

        # Center the window
        self.center_window()

        # Set a custom style for the buttons
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 14))

        ### GUI initialization code ###

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

        # Output Text Box
        self.output_label1 = ttk.Label(root, text="Similar documents:", font=("Helvetica", 16))
        self.output_label1.pack()
        self.output_box1 = tk.Text(root, height=10, width=60, wrap=tk.WORD, font=("Helvetica", 12))
        self.output_box1.pack(padx=10, pady=5)

        # Output Text Box
        self.output_label2 = ttk.Label(root, text="Performance statistics:", font=("Helvetica", 16))
        self.output_label2.pack()
        self.output_box2 = tk.Text(root, height=5, width=60, wrap=tk.WORD, font=("Helvetica", 12))
        self.output_box2.pack(padx=10, pady=5)


def run_gui(doc_dict, retrieval_system):
    root = tk.Tk()
    app = SimpleGUI(root, doc_dict, retrieval_system)
    root.mainloop()
