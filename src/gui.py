import tkinter as tk
from tkinter import ttk


class SimpleGUI:

    def center_window(self, window_width=650, window_height=600):
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
            output_text = f"You entered: {input_text}"
            self.output_box.insert(tk.END, output_text + "\n")
            self.entry.delete(0, tk.END)
        elif selected_option == "Title":
            selected_title = self.title_combobox.get()
            if selected_title:
                output_text = f"You selected the title: {selected_title}"
                self.output_box.insert(tk.END, output_text + "\n")

    def __init__(self, root):
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
        titles = ["Title 1", "Title 2", "Title 3"]

        # Create a title Combobox
        self.title_combobox = ttk.Combobox(root, values=titles, font=("Helvetica", 14), state="readonly")
        self.title_combobox.pack(pady=10)

        # Process Input Button
        self.process_button = ttk.Button(root, text="Process Input", command=self.process_input, style="TButton")
        self.process_button.pack(pady=15)

        # Output Text Box
        self.output_label = ttk.Label(root, text="Output:", font=("Helvetica", 16))
        self.output_label.pack()
        self.output_box = tk.Text(root, height=15, width=60, wrap=tk.WORD, font=("Helvetica", 12))
        self.output_box.pack(padx=10, pady=5)


def run_gui():
    root = tk.Tk()
    app = SimpleGUI(root)
    root.mainloop()
