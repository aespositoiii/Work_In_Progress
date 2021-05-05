from tkinter import *
from PIL import ImageTk, Image  
from tkinter import filedialog
import glob
# from tkmacosx import Button - allows background color and other features not available on macosx

# create the root
root = Tk() 
root.title('Dummy Program')
root.geometry("200x200")


vertical = Scale(root, from_=200, to=600)
vertical.pack()

horizontal = Scale(root, from_=200, to=600, orient=HORIZONTAL)
horizontal.pack()

my_label = Label(root, text=horizontal.get()).pack()

def resize():
    vert = vertical.get()
    horiz = horizontal.get()
    root.geometry(str(horiz) + "x" + str(vert))

resize_button = Button(root, text="RESIZE WINDOW", command=resize)
resize_button.pack()


# run the root
root.mainloop()