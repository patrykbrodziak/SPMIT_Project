from src.app import App


if __name__ == "__main__":
    window = App(900, 684)
    window.load_files("data/cities.csv", "data/citiesaa.csv")
    window.create_listbox()
    window.make_buttons()
    window.run()
