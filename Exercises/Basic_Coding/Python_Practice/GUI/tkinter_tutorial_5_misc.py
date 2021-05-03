from tkinter import *
from PIL import ImageTk, Image
# from tkmacosx import Button - allows background color and other features not available on macosx


# create the root
root = Tk()
root.title('Dummy Program')

# add an icon to the program

# root.iconbitmap(directory + filename + '.ico')

# quit the program button

quit_button = Button(root, text="Exit Program", padx=50, pady=50, command = root.quit)
quit_button.pack()

# Add an Image with Pillow

my_image = ImageTk.PhotoImage(Image.open("/Users/anthonyesposito/Pictures/macroni/Rosasite_w_Conacalcite/1/JPG/stacked/test1619631839.jpg"))
my_label = Label(image=my_image)
my_label.pack()

'''# create a label
myLabel1 = Label(root, text="Hello World!")
myLabel2 = Label(root, text="balogna sales: 1500 units")
myLabel3 = Label(root, text="balogna sales: 1500 units")
myLabel4 = Label(root, text="balogna sales: 1500 units")
myLabel5 = Label(root, text="balogna sales: 1500 units")

# push the label onto the window
myLabel.pack()

# push the label onto the window in grid format
myLabel1.grid(row=0, column=0)
myLabel2.grid(row=1, column=1)
myLabel3.grid(row=2, column=2)
myLabel4.grid(row=3, column=3)
myLabel5.grid(row=4, column=2)
# Entry Widget

e = Entry(root, width=50, bg='grey', borderwidth=12)
e.grid(row=2, column=1)
e.insert(0, 'Default Text')
# make a button


def myClick():
    sometext = Label(root, text=e.get())
    sometext.grid(row=2, column=2)


myButton1 = Button(root, text="BUTTon", state=DISABLED)
myButton1.grid(row=0, column=0)

myButton2 = Button(root, text="Anotha one", padx=5, pady=15, highlightbackground='#000000', fg='red')
myButton2.grid(row=0, column=1)

myButton3 = Button(root, text="Anotha, notha one", command=myClick)
myButton3.grid(row=1, column=2)'''




# run the root
root.mainloop()