import tkinter as tk
from tkinter import ttk

from test import calculatePrecisionAndRecall
from utils import *
from related_doc_retrieval import *
from src import variables


def return_similar_documents(doc_dict, query: str, by_title: bool = False, num_results: int = 10):

    documents = list(doc_dict.values())
    document_titles = list(doc_dict.keys())

    retrieval_system = RelatedDocumentsRetrieval(document_titles, documents)

    # check if tfidf_matrix.csv exists
    retrieval_system._tfidf_matrix = None
    if not os.path.isfile(variables.tfidf_matrix_csv_path):
        retrieval_system._tfidf_matrix = retrieval_system.vectorize_documents()
        store_tfidf_matrix(retrieval_system._tfidf_matrix)
    else:
        retrieval_system._tfidf_matrix = load_tfidf_matrix()
        retrieval_system.own_vectorizer.fit_transform(documents, retrieval_system._tfidf_matrix)

    similar_documents_titles, similar_documents = [], []
    if by_title:
        query_document = doc_dict[query]
        similar_documents_titles, similar_documents, scores = retrieval_system.retrieve_similar_documents(query_document, query, num_results)
    else:
        similar_documents_titles, similar_documents, scores = retrieval_system.retrieve_similar_documents(query, "", num_results)

    print("Query Document:", query)
    print("Similar Documents:")
    for i, doc in enumerate(similar_documents_titles, 1):
        print(f"{i}. {doc}")

    if by_title:
        title = query
        d = read_gt(variables.filepath_path_gt)  # Read ground truth
        print(d[title])

        expected = list(d[title].keys())
        par = calculatePrecisionAndRecall(expected, similar_documents_titles)
        print("fraction of relevant instances among the retrieved instances: ", par[0])
        print("fraction of relevant instances that were retrieved: ", par[1])

        return similar_documents_titles, similar_documents, par[0], par[1]

    return similar_documents_titles, similar_documents


class SimpleGUI:

    def center_window(self, window_width=650, window_height=650):
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

    # def update_input_widget(self, event):
    #     selected_option = self.choice_var.get()
    #     if selected_option == "Query Sentence":
    #         self.title_combobox.pack_forget()
    #         self.entry.pack(pady=10)
    #     elif selected_option == "Title":
    #         self.entry.pack_forget()
    #         self.title_combobox.pack(pady=10)

    def process_input(self):
        selected_option = self.choice_var.get()

        if selected_option == "Query Sentence":
            input_text = self.entry.get()
            output_text = ""
            result = return_similar_documents(self.doc_dict, input_text, by_title=False, num_results=10)
            if len(result) == 4:
                similar_documents_titles, similar_documents, par0, par1 = result
                # output_text1 = f"Query Sentence: {input_text}\n" \
                #               f"Precision: {par0}\n" \
                #               f"Recall: {par1}\n" \
                #               f"Similar Documents:\n"
                output_text1 = f"Similar Documents:\n"
                output_text2 = f"Precision: {par0}\n" \
                               f"Recall: {par1}\n"

                for i, doc in enumerate(similar_documents_titles, 1):
                    output_text1 += f"{i}. {doc}\n"
                self.output_box1.insert(tk.END, output_text1 + "\n")
                self.output_box2.insert(tk.END, output_text2 + "\n")
            else:
                similar_documents_titles, similar_documents = result
                output_text1 = f"Similar Documents:\n"

                for i, doc in enumerate(similar_documents_titles, 1):
                    output_text += f"{i}. {doc}\n"
                self.output_box1.insert(tk.END, output_text1 + "\n")

            # output_text = f"You entered: {input_text}"
            self.output_box1.insert(tk.END, output_text + "\n")
            self.entry.delete(0, tk.END)
        elif selected_option == "Title":
            selected_title = self.title_combobox.get()
            if selected_title:
                output_text = ""
                result = return_similar_documents(self.doc_dict, selected_title, by_title=True, num_results=10)
                if len(result) == 4:
                    similar_documents_titles, similar_documents, par0, par1 = result
                    # output_text1 = f"Title: {selected_title}\n" \
                    #               f"Precision: {par0}\n" \
                    #               f"Recall: {par1}\n" \
                    #               f"Similar Documents:\n"
                    output_text1 = f"Similar Documents:\n"
                    output_text2 = f"Precision: {par0}\n" \
                                   f"Recall: {par1}\n"
                    for i, doc in enumerate(similar_documents_titles, 1):
                        output_text1 += f"{i}. {doc}\n"
                    self.output_box1.insert(tk.END, output_text1 + "\n")
                    self.output_box2.insert(tk.END, output_text2 + "\n")
                else:
                    similar_documents_titles, similar_documents = result
                    output_text1 = f"Similar Documents:\n"
                    for i, doc in enumerate(similar_documents_titles, 1):
                        output_text1 += f"{i}. {doc}\n"
                    self.output_box1.insert(tk.END, output_text1 + "\n")

                # output_text = f"You selected the title: {selected_title}"
                # self.output_box1.insert(tk.END, output_text + "\n")

    def __init__(self, root, doc_dict):

        self.doc_dict = doc_dict
        self.root = root
        root.title("Similar Documents Retrieval System")

        # Center the window
        self.center_window()

        # Set a custom style for the buttons
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 14))

        # GUI initialization code

        # Dropdown for choosing either Query Sentence or Title
        self.choice_var = tk.StringVar()
        self.choice_var.set("Query Sentence")
        self.choice_label = ttk.Label(root, text="Choose an input option:", font=("Helvetica", 16))
        self.choice_label.pack()
        self.choice_combobox = ttk.Combobox(root, values=["Query Sentence", "Title"], textvariable=self.choice_var)
        self.choice_combobox.pack(pady=10)

        # Input Entry
        self.entry_label = ttk.Label(root, text="Input:", font=("Helvetica", 16))
        self.entry_label.pack()
        self.entry = ttk.Entry(root, font=("Helvetica", 14))
        self.entry.pack(pady=10)

        # A list of titles (for testing)
        document_titles = list(doc_dict.keys())
        titles = document_titles

        # Create a title Combobox
        self.title_combobox = ttk.Combobox(root, values=titles, font=("Helvetica", 14), state="readonly")
        self.title_combobox.pack(pady=10)

        # Process Input Button
        self.process_button = ttk.Button(root, text="Process Input", command=self.process_input, style="TButton")
        self.process_button.pack(pady=15)

        # Output Text Box
        self.output_label1 = ttk.Label(root, text="Output:", font=("Helvetica", 16))
        self.output_label1.pack()
        self.output_box1 = tk.Text(root, height=10, width=60, wrap=tk.WORD, font=("Helvetica", 12))
        self.output_box1.pack(padx=10, pady=5)

        # Output Text Box
        self.output_label2 = ttk.Label(root, text="Output:", font=("Helvetica", 16))
        self.output_label2.pack()
        self.output_box2 = tk.Text(root, height=5, width=60, wrap=tk.WORD, font=("Helvetica", 12))
        self.output_box2.pack(padx=10, pady=5)

        # # PanedWindow for two output boxes
        # self.paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        # self.paned_window.pack(expand=True, fill='both')
        #
        # # Output Text Box 1
        # self.output_label1 = ttk.Label(self.paned_window, text="", font=("Helvetica", 16))
        # self.output_label1.pack()
        # self.output_box1 = tk.Text(self.paned_window, height=15, width=40, wrap=tk.WORD, font=("Helvetica", 12))
        # self.output_box1.pack(padx=10, pady=5)
        # self.paned_window.add(self.output_box1)
        #
        # # Output Text Box 2
        # self.output_label2 = ttk.Label(self.paned_window, text="", font=("Helvetica", 16))
        # self.output_label2.pack()
        # self.output_box2 = tk.Text(self.paned_window, height=15, width=20, wrap=tk.WORD, font=("Helvetica", 12))
        # self.output_box2.pack(padx=10, pady=5)
        # self.paned_window.add(self.output_box2)


def run_gui(doc_dict):
    root = tk.Tk()
    app = SimpleGUI(root, doc_dict)
    root.mainloop()
