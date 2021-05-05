from tkinter import *
from PIL import ImageTk, Image  
from tkinter import filedialog
import glob
# from tkmacosx import Button - allows background color and other features not available on macosx

# create the root
root = Tk() 
root.title('Dummy Program')


def open():
    global my_label
    global my_image
    global my_image_label
    root.filename = filedialog.askopenfilename(initialdir="/Users/anthonyesposito/Pictures", title="Select a File", filetypes=(('jpg files', '*.jpg'),('JPG files', '*.JPG'), ('png files', '*.png')))
    my_label = Label(root, text=root.filename)
    my_image = ImageTk.PhotoImage(Image.open(root.filename))
    my_image_label = Label(image=my_image).pack()

my_button = Button(root, padx=20, pady=20, text="Open Image File", command=open)
my_button.pack()

# run the root
root.mainloop()