from tkinter import *
from PIL import ImageTk, Image  
from tkinter import messagebox
# from tkmacosx import Button - allows background color and other features not available on macosx

# create the root
root = Tk()
root.title('Dummy Program')

# Message Box

def popup_info():
    response =  messagebox.showinfo("this is a Popup Window!", "Wassup Everybody")
    Label(root, text="Okie Dokie").pack(padx=15, pady=15)
    '''if response ==1:
        Label(root, text="YES!").pack()
    else:
        Label(root, text="NO!").pack()'''

information = Button(root, text="INFO", padx=15, pady=15, command=popup_info)
information.pack()



def popup_warning():
    response =  messagebox.showwarning("this is a Popup Window!", "Wassup Everybody")
    Label(root, text="Thanks for the Heads Up").pack()

warn = Button(root, text="WARNING", padx=15, pady=15, command=popup_warning)
warn.pack()




def popup_error():
    response =  messagebox.showerror("this is a Popup Window!", "Wassup Everybody")
    Label(root, text="Alright Already!").pack(padx=15, pady=15)

err = Button(root, text="ERROR", padx=15, pady=15, command=popup_error)
err.pack()



def popup_question():
    response =  messagebox.askquestion("this is a Popup Window!", "Wassup Everybody")
    if response == 'yes':
        Label(root, text="YES!").pack()
    elif response == 'no':
        Label(root, text="NO!").pack()

quest = Button(root, text="QUESTION", padx=15, pady=15, command=popup_question)
quest.pack()





def popup_retry():
    response =  messagebox.askretrycancel("this is a Popup Window!", "Wassup Everybody")
    if response == 1:
        Label(root, text="TRYING AGAIN").pack()
    elif response == 0:
        Label(root, text="CANCELING").pack()

retry_button = Button(root, text="RETRY", padx=15, pady=15, command=popup_retry)
retry_button.pack()




def popup_askokcancel():
    response =  messagebox.askokcancel("this is a Popup Window!", "Wassup Everybody")
    if response == 1:
        Label(root, text="PROCEED").pack()
    elif response == 0:
        Label(root, text="CANCEL").pack()

okcancel = Button(root, text="OK/CANCEL", padx=15, pady=15, command=popup_askokcancel)
okcancel.pack()



def popup_yn():
    response =  messagebox.askyesno("this is a Popup Window!", "Wassup Everybody")
    if response == 1:
        Label(root, text="YES!").pack()
    elif response == 0:
        Label(root, text="NO").pack()

yn = Button(root, text="YES/NO", padx=15, pady=15, command=popup_yn)
yn.pack()


def popup_ync():
    response =  messagebox.askyesnocancel("this is a Popup Window!", "Wassup Everybody")
    if response == 1:
        Label(root, text="YES!").pack()
    elif response == 0:
        Label(root, text="NO").pack()
    else:
        Label(root, text="CANCELING").pack()

ync = Button(root, text="YES/NO/CANCEL", padx=15, pady=15, command=popup_ync)
ync.pack()

# run the root
root.mainloop()