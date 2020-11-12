import os
from tkinter import Label, Frame, Scrollbar, VERTICAL, Listbox, MULTIPLE, Button, RIGHT, Y, Tk, filedialog

import pandas as pd

from src.map import Map
from src.solvers import set_vrp_hyper_parameters


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
        self.open_from_file_button = Button(
            self.root, text="OPEN FROM FILE", command=self.open_from_file_on_click_listener
        )
        self.open_from_file_button.pack(pady=10)

        self.show_button = Button(self.root, text="SHOW MAP", command=self.show_on_click_listener)
        self.show_button.pack(pady=10)

        self.find_routes_button = Button(self.root, text="FIND ROUTES", command=self.find_routes_on_click_listener)
        self.find_routes_button.pack(pady=10)
        # data frame
        self.cities_df = pd.DataFrame()
        self.connections = []
        # optimizer parameters
        self.number_of_vehicles = 5
        # map object
        self.map = Map()

    def __call__(self):
        """Run application main window"""
        self.root.mainloop()
        self.root.geometry("400x600")

    def open_from_file_on_click_listener(self) -> None:
        """Load points from file"""
        file = filedialog.askopenfilename(
            initialdir=os.getcwd(), title="Open Data File", filetypes=(("all files", "*"),)
        )
        self.cities_df = pd.read_csv(file)

    def show_on_click_listener(self):
        """Show map with loaded points"""
        self.map(self.cities_df, self.connections)

    def find_routes_on_click_listener(self):
        self.find_route()

    def find_route(self):
        optimizer = set_vrp_hyper_parameters(n_points=self.cities_df.index.size, n_agents=self.number_of_vehicles)
        route, length = optimizer.minimize(
            self.cities_df[["lat", "long"]].values, num_steps=100, patience=10, silent=True
        )

        self.connections = route

    def set_scrollbar_position(self) -> None:
        """Set initial scrollbar position"""
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.frame.pack()
