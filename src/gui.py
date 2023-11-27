import tkinter as tk
from tkinter import ttk


class SimpleGUI:

    def center_window(self, window_width=600, window_height=400):
        """
        Center the window on the screen.
        :param window_width: window width
        :param window_height: window height
        :return:
        """
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        # calculate position x and y coordinates
        x = (screen_width/2) - (window_width/2)
        y = (screen_height/2) - (window_height/2)
        self.master.geometry('%dx%d+%d+%d' % (window_width, window_height, x, y))

    def __init__(self, master):
        self.master = master
        master.title("Simple GUI")

        # Center the window
        self.center_window()

        # Set a custom style for the buttons
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 14))

        # GUI initialization code

        # Input Entry
        self.entry_label = ttk.Label(master, text="Input:", font=("Helvetica", 16))
        self.entry_label.pack()
        self.entry = ttk.Entry(master, font=("Helvetica", 14))
        self.entry.pack(pady=10)

        # Process Input Button
        self.process_button = ttk.Button(master, text="Process Input", command=self.process_input, style="TButton")
        self.process_button.pack(pady=15)

        # Output Text Box
        self.output_label = ttk.Label(master, text="Output:", font=("Helvetica", 16))
        self.output_label.pack()
        self.output_box = tk.Text(master, height=10, width=60, wrap=tk.WORD, font=("Helvetica", 12))
        self.output_box.pack(padx=10, pady=5)

    def process_input(self):
        input_text = self.entry.get()
        output_text = f"You entered: {input_text}"
        # Insert the processed output into the text widget
        self.output_box.insert(tk.END, output_text + "\n")
        # Clear the input entry
        self.entry.delete(0, tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleGUI(root)
    root.mainloop()
