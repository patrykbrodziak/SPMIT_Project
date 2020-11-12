import os
import csv
from tkinter import Label, Frame, Scrollbar, VERTICAL, Listbox, MULTIPLE, Button, END, filedialog, RIGHT, Y, Tk

import pandas as pd

from src.map import Map


class App:
    """Main window class"""
    def __init__(self, width: int, height: int):
        """
        :param width: width in pixels
        :param height: height in pixels
        """
        self.root = Tk()
        self.root.geometry("400x600")

        self.width = width
        self.height = height
        self.my_label = Label(self.root)
        self.my_label.pack(pady=80)
        self.frame = Frame(self.root)
        self.scrollbar = Scrollbar(self.frame, orient=VERTICAL)
        self.my_listbox = Listbox(self.frame, width=50, yscrollcommand=self.scrollbar.set, selectmode=MULTIPLE)

    def load_files(self, city_file_path, connections_file_path):
        cities = pd.read_csv(city_file_path)
        self.my_list = [cities['lat'], cities['long'], cities['city']]
        connects = pd.read_csv(connections_file_path)
        self.lat_dict = {}

        for i in range(len(self.my_list[2])):
            self.lat_dict[self.my_list[2][i]] = self.my_list[0][i]
        self.long_dict = {}
        for i in range(len(self.my_list[2])):
            self.long_dict[self.my_list[2][i]] = self.my_list[1][i]

    def run(self):
        """

        """
        self.root.mainloop()
        self.root.geometry("400x600")


    # def make_background(self):
    #     self.canvas = Canvas(root, width=self.width, height=self.height)
    #     self.filename = PhotoImage(file=r"grafika.gif")
    #     background_label = Label(root, image=self.filename)
    #     background_label.place(relwidth=1, relheight=1)
    #     self.canvas.pack()

    def make_buttons(self):
        """Create buttons in app"""
        self.select_button = Button(self.root, text="SELECT", command=self.select_all)
        self.select_button.pack(pady=10)
        self.open_from_file_button = Button(self.root, text="OPEN FROM FILE", command=self.open_csv_file)
        self.open_from_file_button.pack(pady=10)

    def create_listbox(self) -> None:
        """

        """
        self.scrollbar_position()
        self.my_listbox.pack(padx=20, pady=10)
        for item in self.my_list[2]:
            self.my_listbox.insert(END, item)

    def select_all(self):
        result = ""
        for item in self.my_listbox.curselection():
            result = result + str(self.my_listbox.get(item)) + '\n'
        self.my_label.config(text=result)
        self.result_list = result.split('\n')
        self.result_list = self.result_list[0:-1]
        self.save_to_csv()
        map = Map('result.csv')

    def open_csv_file(self):
        text_file = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Open Text File",
            filetypes=(("all files", "*"), )
        )

        map = Map(text_file)

    def scrollbar_position(self):
        self.scrollbar.config(command=self.my_listbox.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.frame.pack()

    def save_to_csv(self):
        with open('result.csv', 'w', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['lat', 'long', 'city'])
            for city in self.result_list:
                csv_writer.writerow([self.lat_dict[city], self.long_dict[city], city])
