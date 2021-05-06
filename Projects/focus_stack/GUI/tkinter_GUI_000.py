from tkinter import *
from PIL import ImageTk, Image  
from tkinter import messagebox
from tkinter import filedialog
import glob
import cv2 as cv
# from tkmacosx import Button - allows background color and other features not available on macosx

# create the root
root = Tk() 
root.title('Image Processing Application')
root.geometry("200x400")

# Function def to open a new window

def proc_dev_interface():

    proc_dev_win.geometry("800x1000")

    # Build the filter selection frame
    filters = [ ("Gabor", 0),
                ("Gauss", 1),
                ("Canny", 2),
                ("Laplace", 3),
                ("FFT", 4)]

    filter_buttons = [[]]

    for i in range(len(filters))

    filter_frame = LabelFrame(proc_dev_win, text = "Filter Selection")
    filter_frame.place(x=40, width=360, y=40, height=500)

    Gabor_Rad = Radiobutton(filter_frame, text="Gabor")
    Gabor_Rad.place(relx = .1, rely=.1)

    Gauss_Rad = Radiobutton(filter_frame, text="Gauss")
    Gauss_Rad.place(relx = .1, rely=.3)

    Canny_Rad = Radiobutton(filter_frame, text="Canny")
    Canny_Rad.place(relx = .1, rely=.5)

    Laplace_Rad = Radiobutton(filter_frame, text="Laplace")
    Laplace_Rad.place(relx = .1, rely=.7)

    FFT_Rad = Radiobutton(filter_frame, text="FFT")
    FFT_Rad.place(relx = .1, rely=.9)



# Open the file for process development

def open_proc_dev_file():

    global image1

    root.filename = filedialog.askopenfilename(initialdir="/Users/anthonyesposito/Pictures", title="Select a File", filetypes=(('jpg files', '*.jpg'),('JPG files', '*.JPG'), ('png files', '*.png')))
    image1 = cv.imread(root.filename, cv.IMREAD_COLOR)
    cv.imshow('Base Image', image1)

    proc_dev_interface()

    proc_dev_button.place_forget()

# Open the winodw for process development

def open_proc_dev():
    

    global proc_dev_win

    dev_im_proc_button["state"] = "disabled"
    batch_proc_button["state"] = "disabled"

    proc_dev_win = Toplevel()
    proc_dev_win.geometry("300x40")
    proc_dev_win.title("Image Processing Development")

    global proc_dev_button
    proc_dev_button = Button(proc_dev_win, padx=5, pady=5, text="Select Base Image", command=open_proc_dev_file)
    proc_dev_button.place(x=10, width =280, y=10, height=20)
    
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to leave process development?"):
            if messagebox.askyesno("Save Work?", "Would you like to save your progress?"):
                print("Save Progress")
            proc_dev_win.destroy()
            cv.destroyWindow("Base Image")
            dev_im_proc_button["state"] = "normal"
            batch_proc_button["state"] = "normal"


    proc_dev_win.protocol("WM_DELETE_WINDOW", on_closing)

# Open the batch processing window
def open_batch_proc():

    batch_proc_win = Toplevel()
    batch_proc_win.title("Batch Processing")



# Set up the action buttons on the main window

dev_im_proc_button = Button(root, padx=20, pady=20, text="Develop Image Process", command=open_proc_dev)
dev_im_proc_button.place(relx=.1, relwidth=.8, rely=.3, relheight=.2)

batch_proc_button = Button(root, padx=20, pady=20, text="Batch Processing", command=open_batch_proc)
batch_proc_button.place(relx=.1, relwidth=.8, rely=.55, relheight=.2)

quit_button = Button(root, padx=20, pady=20, text="Quit", command=root.quit)
quit_button.place(relx=.1, relwidth=.8, rely=.8, relheight=.15)

def on_closing():
    if messagebox.askokcancel("Quit", "Are you sure you would like to quit?"):
        print("Save Progress")
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# run the root
root.mainloop()