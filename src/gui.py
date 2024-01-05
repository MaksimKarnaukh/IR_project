import tkinter as tk
from tkinter import ttk

from test import calculatePrecisionAndRecall
from utils import *
from related_doc_retrieval import *
from src import variables


class SimpleGUI:

    def return_similar_documents(self, doc_dict, query: str, by_title: bool = False, num_results: int = 10):

        similar_documents_titles, similar_documents = [], []
        if by_title:
            query_document = doc_dict[query]
            similar_documents_titles, similar_documents, scores = self.retrieval_system.retrieve_similar_documents(
                query_document, query, num_results)
        else:
            similar_documents_titles, similar_documents, scores = self.retrieval_system.retrieve_similar_documents(query, "",
                                                                                                              num_results)
        if by_title:
            title = query
            d = read_gt(variables.filepath_path_gt)  # Read ground truth
            print(d[title])

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
            # check if self.num_results_entry.get() is a number in string format
            num_results = 10
            if self.num_results_entry.get().isdigit():
                num_results = int(self.num_results_entry.get())
            result = self.return_similar_documents(self.doc_dict, input_text, by_title=False, num_results=num_results)
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
                    output_text1 += f"{i}. {doc}\n"
                self.output_box1.insert(tk.END, output_text1 + "\n")

            # output_text = f"You entered: {num_results}"
            # self.output_box1.insert(tk.END, output_text + "\n")

            self.entry.delete(0, tk.END)

        elif selected_option == "Title":
            selected_title = self.title_combobox.get()
            if selected_title:
                num_results = 10
                if self.num_results_entry.get().isdigit():
                    num_results = int(self.num_results_entry.get())
                result = self.return_similar_documents(self.doc_dict, selected_title, by_title=True, num_results=num_results)
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

        # Additional Input for Number of Results
        self.num_results_label = ttk.Label(root, text="Number of Results:", font=("Helvetica", 16))
        self.num_results_label.pack()
        self.num_results_entry = ttk.Entry(root, font=("Helvetica", 14))
        self.num_results_entry.pack(pady=10)

        # A list of titles
        titles = self.document_titles

        # Create a title Combobox
        self.title_combobox = ttk.Combobox(root, values=titles, font=("Helvetica", 14), state="readonly")
        self.title_combobox.pack(pady=10)

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


def run_gui(doc_dict, retrieval_system):
    root = tk.Tk()
    app = SimpleGUI(root, doc_dict, retrieval_system)
    root.mainloop()
