import pandas as pd
import plotly.graph_objects as go
from tkinter import *
from PIL import ImageTk, Image
import csv
from tkinter import filedialog

root = Tk()
root.geometry("400x600")
cities = pd.read_csv('cities.csv')
my_list = [cities['lat'], cities['long'], cities['city']]
connects = pd.read_csv('citiesaa.csv')
lat_dict = {}
for i in range(len(my_list[2])):
    lat_dict[my_list[2][i]] = my_list[0][i]
long_dict = {}
for i in range(len(my_list[2])):
    long_dict[my_list[2][i]] = my_list[1][i]



class App():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.my_label = Label(root)
        self.my_label.pack(pady=80)
        self.frame = Frame(root)
        self.scrollbar = Scrollbar(self.frame, orient=VERTICAL)
        self.my_listbox = Listbox(self.frame, width=50, yscrollcommand=self.scrollbar.set, selectmode=MULTIPLE)


    # def make_background(self):
    #     self.canvas = Canvas(root, width=self.width, height=self.height)
    #     self.filename = PhotoImage(file=r"grafika.gif")
    #     background_label = Label(root, image=self.filename)
    #     background_label.place(relwidth=1, relheight=1)
    #     self.canvas.pack()

    def make_buttons(self):
        self.select_button = Button(root, text="SELECT", command=self.select_all)
        self.select_button.pack(pady=10)
        self.select_button = Button(root, text="OPEN FROM FILE", command=self.open_csv_file)
        self.select_button.pack(pady=10)

    def create_listbox(self, list):
        self.scrollbar_position(self)
        self.my_listbox.pack(padx=20, pady=10)
        for item in list[2]:
            self.my_listbox.insert(END, item)

    def select_all(self):
        result = ''
        for item in self.my_listbox.curselection():
            result = result + str(self.my_listbox.get(item)) + '\n'
        self.my_label.config(text=result)
        self.result_list = result.split('\n')
        self.result_list = self.result_list[0:-1]
        self.save_to_csv(self)
        map = Map('result.csv')

    def open_csv_file(self):
        text_file = filedialog.askopenfilename(initialdir="C:\\Users\\patry\\PycharmProjects\\aaa", title="Open Text File", filetypes=(("all files", "*"), ))
        print(text_file)
        map = Map(text_file)

    @staticmethod
    def scrollbar_position(self):
        self.scrollbar.config(command=self.my_listbox.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.frame.pack()

    @staticmethod
    def save_to_csv(self):
        with open('result.csv', 'w', encoding='utf-8') as kupa:
            csvwriter = csv.writer(kupa)
            csvwriter.writerow(['lat', 'long', 'city'])
            for city in self.result_list:
                csvwriter.writerow([lat_dict[city], long_dict[city], city])
class Map():
    def __init__(self, csv_file):
        self.cities2 = pd.read_csv(csv_file)
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            lon = self.cities2['long'],
            lat = self.cities2['lat'],
            hoverinfo = 'text',
            text = self.cities2['city'],
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgb(255, 0, 0)',
            )))
        # for i in range(len(connects)):
        #     fig.add_trace(
        #         go.Scattermapbox(
        #             lon = [connects['start_lon'][i], connects['end_lon'][i]],
        #             lat = [connects['start_lat'][i], connects['end_lat'][i]],
        #             mode = 'lines',
        #             text = cities['city'],
        #         )
        #     )
        fig.update_layout(
            showlegend = False,
            geo = dict(
                scope = 'north america',
                projection_type = 'azimuthal equal area',
                showland = True,
                landcolor = 'rgb(243, 243, 243)',
                countrycolor = 'rgb(204, 204, 204)',
            ),
        )
        fig.update_layout(
                margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                mapbox={
                    'center': {'lon': 20, 'lat': 50},
                    'style': "open-street-map",
                    'zoom': 5})
        fig.show()

window = App(900, 684)
# window.make_background()
window.create_listbox(my_list)
window.make_buttons()
root.mainloop()















