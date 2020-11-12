from pathlib import Path
from tkinter import Label, Frame, Scrollbar, VERTICAL, Listbox, MULTIPLE, Button, END, RIGHT, Y, Tk
from typing import Union

import pandas as pd

from src.map import Map


class App:
    """Main window class"""
    def __init__(self, width: int, height: int):
        """
        :param width: width in pixels
        :param height: height in pixels
        """
        # Global app window
        self.root = Tk()
        self.root.geometry("400x600")
        # labels and size
        self.width = width
        self.height = height
        self.main_label = Label(self.root)
        self.main_label.pack(pady=80)
        self.frame = Frame(self.root)
        self.scrollbar = Scrollbar(self.frame, orient=VERTICAL)
        self.listbox = Listbox(self.frame, width=50, yscrollcommand=self.scrollbar.set, selectmode=MULTIPLE)
        # buttons
        self.select_button = Button(self.root, text="SELECT", command=self.select_on_click_listener)
        self.select_button.pack(pady=10)
        self.open_from_file_button = Button(
            self.root, text="OPEN FROM FILE", command=self.open_from_file_on_click_listener
        )
        self.open_from_file_button.pack(pady=10)
        # data frame
        self.cities_df = pd.DataFrame()
        self.connections_df = pd.DataFrame()
        # map object
        self.map = Map()

    def __call__(self):
        """Run application main window"""
        self.root.mainloop()
        self.root.geometry("400x600")

    def load_files(self, city_file_path: Union[str, Path], connections_file_path: Union[str, Path]) -> None:
        self.cities_df = pd.read_csv(city_file_path)
        self.connections_df = pd.read_csv(connections_file_path)

    def create_listbox(self) -> None:
        """Build list box from loaded data"""
        self.set_scrollbar_position()
        self.listbox.pack(padx=20, pady=10)

        for item in self.cities_df["city"].tolist():
            self.listbox.insert(END, item)

    def select_on_click_listener(self) -> None:
        """Show map with loaded points on click"""
        self.map(self.cities_df)

    def open_from_file_on_click_listener(self) -> None:
        """Show map with points loaded from file"""
        self.map(self.cities_df)

    def set_scrollbar_position(self) -> None:
        """Set initial scrollbar position"""
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.frame.pack()
