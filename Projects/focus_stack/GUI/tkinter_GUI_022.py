from tkinter import *
from PIL import ImageTk, Image  
from tkinter import messagebox
from tkinter import filedialog
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from log_gabor.log_gabor import log_gabor
import os
from datetime import datetime
from aesop import aesop
from skimage.transform import resize 
import json
import time
from process_and_stack.process_and_stack import Stacking

from numpy.lib.npyio import save
# from tkmacosx import Button - allows background color and other features not available on macosx

def plot_hist(image, title):
    hist_figure = plt.figure(title)
    plt.clf()
    plt.title(title)
    histogram = np.histogram(image.ravel(), bins=256, range=[0,256])
    plt.yscale('log')
    plt.plot(histogram[1][0:256], histogram[0])
    plt.xlim(0, 255)
    plt.ylim(1, histogram[0].max())
    plt.yscale('log')
    plt.draw()
    plt.show(block=False)


# create the root
root = Tk() 
root.title('Image Processing Application')
root.geometry("200x400")

# Function def to open a new window

def proc_dev_interface():

    global process_ops_dict

    proc_dev_win = Toplevel()
    proc_dev_win.title("Process Development")
    proc_dev_win.geometry("800x1000")
    
    def on_process_closing():
        if messagebox.askokcancel("Leave Process Development", "Do you want to leave process development?"):
            if len(images) > 1:
                if messagebox.askyesno("Save Work?", "Would you like to save your progress?"):
                    save_process()
            proc_dev_win.destroy()
            cv.destroyAllWindows()
            dev_im_proc_button["state"] = "normal"
            batch_proc_button["state"] = "normal"
    plt.close('all')

    proc_dev_win.protocol("WM_DELETE_WINDOW", on_process_closing)
######################################          Build the process options             ######################################
    
    
    def process_options():
        
        #   Declare global variables for use in other functions

        global parameter_list
        global process_ops_frame
        global process_ops_dict
        global image_source
        global math_type
        global thresh_select
        global threshold_option
        global norm_select
        global norm_option
        global math_routine
        global save_mask_button
        global save_process
        global morph_select
        global truncate
        global misc_select

        #   Clear the operation frame when a new process is selected

        try:
            process_ops_frame.place_forget()
            #plt.close()
        except:
            pass
        
        
        #   Generate the operations frame and add the optionMenu for the image to be operated on.

        process_ops_frame = LabelFrame(proc_dev_win, text = (process_select.get() + " Process Options"))
        process_ops_frame.place(x=410, width=370, y=40, height=600)
        image_select_src = StringVar(value=image_names[0])
        Label(process_ops_frame, text='Source Image Selection:', anchor='w').place(x=20, y=10)
        image_source = OptionMenu(process_ops_frame, image_select_src, *image_names)
        image_source.place(x=200, y=10)

        # The process_ops_dict is a dictionary that provides the parameters for generating the scales and options for each selected process
        # For Options utilizing scales for parameter selection:
        # ( ... , "process NAME" : [ 'PARAMETER NAME', ' - PARAMETER DESCRIPTION', SCALE MIN, SCALE MAX, RESOLUTION])


        filter_list = [ "Gabor",
                        "Log_Gabor",
                        "Gauss",
                        "Canny",
                        "Laplace",
                        "Aesop"]

        # Generate the scales for the filters
        if process_select.get() in filter_list:
            parameter_list = process_ops_dict[process_select.get()]
            for parameters in parameter_list:
                exec( "global {}_scale".format(parameters[0]), globals())
                exec( "{}_scale = Scale(process_ops_frame, label = '{}', from_= {}, to= {}, resolution = {}, orient= HORIZONTAL, length=300)".format((parameters[0]), (parameters[0]+parameters[1]), parameters[2], parameters[3], parameters[4]), globals())
                exec( "{}_scale.place(x=10, y=40+{}*60)".format(parameters[0], parameter_list.index(parameters)))

            if process_select.get() == "Log_Gabor":
                
                def trunc_func():
                    global truncate_scale
                    if truncate.get() == True:
                        
                        truncate_scale = Scale(process_ops_frame, label = 'Truncate Limit i - ( 10^i )', from_=1, to=6, resolution=.1, orient=HORIZONTAL, length=300)
                        truncate_scale.place(x=20, y=80+len(parameter_list)*70)

                    elif truncate.get() == False:
                        truncate_scale.place_forget()
                        
                truncate = BooleanVar()
                truncate_histogram = Checkbutton(process_ops_frame, text="Truncate histogram?", variable=truncate, command=trunc_func, onvalue=True, offvalue=False)
                truncate_histogram.place(x=20, y=50+len(parameter_list)*70)

                LG_normalize = BooleanVar()
                LG_normalize_histogram = Checkbutton(process_ops_frame, text="Normalize histogram?", variable=LG_normalize, onvalue=True, offvalue=False)
                LG_normalize_histogram.place(x=20, y=20+len(parameter_list)*70)
        # Generate the interface for math operations.  Math Operations are broken into 
        # Arithmetic, comparisons, and logs and exponents.

        

        
        elif process_select.get() == "Math":
                def math_routine():

                    global math_routine_frame
                    global const
                    global second_select_src
                    global math_operation
                    global math_constant
                    global set_operators

                    # set_operators populates the options for each category of math operation and the 
                    # secondary operands for each operation if applicable.

                    def set_operators():
                        global const
                        global math_constant
                        global second_select_src
                        global math_operation



                        if (math_type.get() == 'Arithmetic') | (math_type.get() == 'Compare'):
                            
                            global math_op
                            # For operations with secondary operands the option for a constant operator or a
                            # previously generated image matrix is presented.
                            
                            math_op = math_operation.get()

                            def use_constant():
                                global math_constant
                                
                                

                                if const.get() == True:
                                    second_source_label.place_forget()
                                    image_source2.place_forget()
                                    math_constant.place(x=20, y=50+len(operations)*30)

                                elif const.get() == False:
                                    math_constant.place_forget()
                                    second_source_label.place(x=20, y=50+len(operations)*30)
                                    image_source2.place(x=200, y=50+len(operations)*30)
                                    
                            second_source_label = Label(math_routine_frame, text='Secondary Source Selection:', anchor='w')
                            second_source_label.place(x=20, y=50+len(operations)*30)
                            second_select_src = StringVar(value=image_names[0])
                            image_source2 = OptionMenu(math_routine_frame, second_select_src, *image_names)
                            image_source2.place(x=200, y=50+len(operations)*30)
                            
                            if math_operation.get() == 'Multiply':
                                math_constant = Scale(math_routine_frame, label='Constant Operator', from_=.05, to=10, resolution=.05, orient=HORIZONTAL, length=300)
                            #math_constant.place(x=20, y=90+len(operations)*30)
                            else:
                                math_constant = Scale(math_routine_frame, label='Constant Operator', from_=1, to=255, resolution=1, orient=HORIZONTAL, length=300)

                            const = BooleanVar()
                            const.set(False)
                            is_constant = Checkbutton(math_routine_frame, text="Use Constant?", variable=const, command=use_constant, onvalue=True, offvalue=False)
                            is_constant.place(x=20, y=20+len(operations)*30)



                        elif math_operation.get() == '(im)^n':

                            math_constant = Scale(math_routine_frame, label='Constant Exponent', from_=1, to=4, resolution=1, orient=HORIZONTAL, length=300)
                            math_constant.place(x=20, y=90+len(operations)*30)

                        elif math_type.get() == 'Logs_and_Exponents':
                            try:
                                math_constant.place_forget()
                            except:
                                pass
                    math_routine_frame = LabelFrame(process_ops_frame, text = (math_type.get() + " Options"), height=20, width=20)
                    math_routine_frame.place(x=10, y=150, height=420, width=340)

                    math_operation = StringVar()
                    math_operation.set('N/A')
                    operations = process_ops_dict["Math"][1][math_type.get()]

                    for i in range(len(operations)):
                        Radiobutton(math_routine_frame, text=operations[i], variable=math_operation, command=set_operators, anchor='w', value=operations[i]).place(x=20, y=10+i*30)
                    

                    

                    

                math_type = StringVar()
                math_type.set('N/A')
                math_option = process_ops_dict["Math"][0]
                
                for i in range(len(math_option)):
                    Radiobutton(process_ops_frame, text=math_option[i], variable=math_type, command=math_routine, anchor='w', value = math_option[i]).place(x=20, y=40+i*30)



        # Generate the options for thresholding filters.
          
        elif process_select.get() == "Thresholding":
            
            def thresh_routine():
                global thresh_ops_frame
                global thresh_select
                global parameter_list
                global trunc_hist_scale

                try:
                    thresh_ops_frame.place_forget()
                    #plt.close()
                except:
                    pass
                
                thresh_ops_frame = LabelFrame(process_ops_frame, text = (process_select.get() + " Process Options"))
                thresh_ops_frame.place(x=10, y=50+len(threshold_option)*30, width=340, height=220)

                thresh_max = ['Binary', 'Inverse_Binary', 'Truncated', 'To_Zero', 'Inverse_To_Zero', 'Otsu_Bin', 'Triangle_Bin']
                adapt_thresh = ['Adaptive_Thresh_Mean_C', 'Adaptive_Thresh_Gaussian_C']

                if thresh_select.get() in thresh_max:
                    parameter_list = process_ops_dict["Thresholding"][1][0:2]
                    for parameters in parameter_list:
                        exec( "global {}_scale".format(parameters[0]), globals())
                        exec( "{}_scale = Scale(thresh_ops_frame, label = '{}', from_= {}, to= {}, resolution = {}, orient= HORIZONTAL, length=300)".format((parameters[0]), (parameters[0]+parameters[1]), parameters[2], parameters[3], parameters[4]), globals())
                        exec( "{}_scale.place(x=10, y=10 + {}*60)".format(parameters[0], parameter_list.index(parameters)))
                
                elif thresh_select.get() in adapt_thresh:
                    parameter_list = process_ops_dict["Thresholding"][1][1:4]
                    for parameters in parameter_list:
                        exec( "global {}_scale".format(parameters[0]), globals())
                        exec( "{}_scale = Scale(thresh_ops_frame, label = '{}', from_= {}, to= {}, resolution = {}, orient= HORIZONTAL, length=300)".format((parameters[0]), (parameters[0]+parameters[1]), parameters[2], parameters[3], parameters[4]), globals())
                        exec( "{}_scale.place(x=10, y=10 + {}*60)".format(parameters[0], parameter_list.index(parameters)))        

                elif thresh_select.get() == 'Trunc_Hist':
                    trunc_hist_scale = Scale(thresh_ops_frame, label = 'Trunc Threshold - histogram count cutoff', from_= .1, to=7, resolution = .1, orient= HORIZONTAL, length=300)
                    trunc_hist_scale.place(x=5, y=10)

            threshold_option = process_ops_dict["Thresholding"][0]

            
            thresh_select = StringVar()
            thresh_select.set('N/A')

            for i in range(len(threshold_option)):
                Radiobutton( process_ops_frame, text=threshold_option[i], variable=thresh_select, command=thresh_routine, anchor='w', value = threshold_option[i]).place(x=20, y=40+i*30)
        


        # Generate the options for the normalize filters.
                
        elif process_select.get() == "Normalize":
            
            def norm_routine():
                global norm_ops_frame
                global norm_select
                global parameter_list
                

                try:
                    norm_ops_frame.place_forget()
                    #plt.close()
                except:
                    pass
                
                norm_ops_frame = LabelFrame(process_ops_frame, text = (process_select.get() + " Process Options"))
                norm_ops_frame.place(x=10, y=70+len(norm_option)*30, width=340, height=230)

                parameter_list = process_ops_dict["Normalize"][1]
                for parameters in parameter_list:
                    exec( "global {}_scale".format(parameters[0]), globals())
                    exec( "{}_scale = Scale(norm_ops_frame, label = '{}', from_= {}, to= {}, resolution = {}, orient= HORIZONTAL, length=300)".format((parameters[0]), (parameters[0]+parameters[1]), parameters[2], parameters[3], parameters[4]), globals())
                    exec( "{}_scale.place(x=10, y=10 + {}*60)".format(parameters[0], parameter_list.index(parameters)))
            
            norm_option = process_ops_dict["Normalize"][0]
            
            norm_select = StringVar()
            norm_select.set('N/A')

            for i in range(len(norm_option)):
                Radiobutton( process_ops_frame, text=norm_option[i], variable=norm_select, command=norm_routine, anchor='w', value = norm_option[i]).place(x=20, y=40+i*30)
        
        
        # Generate the options for the Miscellaneous filters.
                
        elif process_select.get() == "Miscellaneous":
            
            def misc_routine():
                global misc_ops_frame
                global misc_select
                global parameter_list
                global resize_select
                global resize_options
                global second_select_src
                

                try:
                    misc_ops_frame.place_forget()
                    #plt.close()
                except:
                    pass
                
                if misc_select.get() == 'Resize':
                    misc_ops_frame = LabelFrame(process_ops_frame, text = (misc_select.get() + " - Reduce or Match Size"))
                    misc_ops_frame.place(x=10, y=70+len(misc_option)*30, width=340, height=200)

                    def resize_routine():
                        
                        global resize_select
                        global resize_options
                        global second_select_src
                        global scale_scale

                        try:
                            resize_options_frame.place_forget()
                            #plt.close()
                        except:
                            pass

                        resize_options_frame = LabelFrame(misc_ops_frame, text = "Set Reduction Scale")
                        resize_options_frame.place(x=5, y=10+len(misc_option)*30, width=325, height=100)
                        
                        if resize_select.get() == 'Reduce':
                            scale_scale = Scale(resize_options_frame, label = 'Scale - height and width reduction', from_= .05, to= 1, resolution = .05, orient= HORIZONTAL, length=300)
                            scale_scale.place(x=5, y=10)

                        elif resize_select.get() == 'Match_Size':
                            second_source_label = Label(resize_options_frame, text='Secondary Source Selection:', anchor='w')
                            second_source_label.place(x=5, y=10)
                            second_select_src = StringVar(value=image_names[0])
                            image_source2 = OptionMenu(resize_options_frame, second_select_src, *image_names)
                            image_source2.place(x=195, y=10)

                resize_options = ['Reduce', 'Match_Size']
                
                resize_select = StringVar()
                resize_select.set('N/A')

                for i in range(len(resize_options)):
                    Radiobutton( misc_ops_frame, text=resize_options[i], variable=resize_select, command=resize_routine, anchor='w', value = resize_options[i]).place(x=10, y=10+i*30)
                    
                
            misc_option = process_ops_dict["Miscellaneous"][0]
            
            misc_select = StringVar()
            misc_select.set('N/A')

            for i in range(len(misc_option)):
                Radiobutton( process_ops_frame, text=misc_option[i], variable=misc_select, command=misc_routine, anchor='w', value = misc_option[i]).place(x=20, y=40+i*30)


        elif process_select.get() == "Morphology":
            
            def morph_routine():
                global morph_ops_frame
                global parameter_list
                global morph_kernel_select
                global morph_select
                

                try:
                    morph_ops_frame.place_forget()
                    #plt.close()
                except:
                    pass
                
                morph_ops_frame = LabelFrame(process_ops_frame, text = (process_select.get() + " Process Options"))
                morph_ops_frame.place(x=10, y=30+len(morph_option)*30, width=340, height=340)

                morph_kernel_option = process_ops_dict["Morphology"][1]
                
                morph_kernel_select = StringVar()
                morph_kernel_select.set('N/A')

                for i in range(len(morph_kernel_option)):
                    Radiobutton( morph_ops_frame, text=morph_kernel_option[i], variable=morph_kernel_select, anchor='w', value = morph_kernel_option[i]).place(x=20, y=10+i*30)

                if (morph_select.get() == 'Erode') | (morph_select.get() == 'Dilate'):
                    parameter_list = process_ops_dict["Morphology"][2]

                else:
                    parameter_list = process_ops_dict["Morphology"][2][:2]
                
                for parameters in parameter_list:
                    exec( "global {}_scale".format(parameters[0]), globals())
                    exec( "{}_scale = Scale(morph_ops_frame, label = '{}', from_= {}, to= {}, resolution = {}, orient= HORIZONTAL, length=300)".format((parameters[0]), (parameters[0]+parameters[1]), parameters[2], parameters[3], parameters[4]), globals())
                    exec( "{}_scale.place(x=10, y=120 + {}*60)".format(parameters[0], parameter_list.index(parameters)))
            
            morph_option = process_ops_dict["Morphology"][0]
            
            morph_select = StringVar()
            morph_select.set('N/A')

            for i in range(len(morph_option)):
                Radiobutton( process_ops_frame, text=morph_option[i], variable=morph_select, command=morph_routine, anchor='w', value = morph_option[i]).place(x=20, y=40+i*30)
        



#######################################      Generate the filtered image previews     ###########################################
       

        def preview_result():
            global parameter_values
            global parameter_list
            global math_type
            global math_operation
            global const
            global math_constant
            global second_select_src
            global result
            global operand2
            global morph_select
            global morph_kernel_select
            global truncate
            global scaled_size
            
            t1 = time.time()

            apply_button['state'] = 'normal'

            image_selection = images[image_names.index(image_select_src.get())]

            result = np.zeros(image_selection.shape, image_selection.dtype)

            use_parameters = ["Gabor", "Log_Gabor", "Gauss", "Canny", "Laplace", "Aesop", "Thresholding", "Normalize", "Morphology"]


            if process_select.get() in use_parameters:
                parameter_values = [[]] * len(parameter_list)
                for i in range(len(parameter_list)):
                    exec('parameter_values[{}] = {}_scale.get()'.format(i,parameter_list[i][0]), globals())


            if process_select.get() == "Gabor":
                
                '''
                try:
                    plt.close()
                except:
                    pass
                '''

                kernel = cv.getGaborKernel((parameter_values[0], parameter_values[0]), parameter_values[1], parameter_values[2] * np.pi, parameter_values[3] * np.pi, parameter_values[4], parameter_values[5] * np.pi, ktype=cv.CV_32F)
                
                kernel_figure = plt.figure('Kernel Preview')
                plt.clf()
                plt.title('Kernel')
                plt.imshow(kernel)
                plt.draw()
                plt.show(block=False)
                
                result = cv.filter2D(image_selection, cv.CV_8U, kernel)
                print(result.dtype)


            elif process_select.get() == "Log_Gabor":

                result, LG = log_gabor(image_selection, parameter_values[0], parameter_values[1], parameter_values[2], parameter_values[3])
                histogram = np.histogram(result.ravel(), bins=256, range=[0,256])
                if LG_normalize.get() == True:
                    peak = (np.where(histogram[0] == np.max(histogram[0])))[0][0]
                    if peak <= 127:
                        result[result < peak] = peak + (peak - result[result < peak])
                    elif peak > 127:
                        result[result > peak] = peak - (result[result > peak] - peak)
                        result = (255 * np.ones(result.shape, dtype='uint8')) - result                    

                if truncate.get() == True:
                    zeroed = np.where([histogram[0] > 10**truncate_scale.get()])[1]
                    
                    for j in zeroed:
                        result[result==j]=0

                cv.imshow('Mask Preview', LG)


            elif process_select.get() == "Gauss":

                result = cv.GaussianBlur(image_selection, (parameter_values[0], parameter_values[0]), 0)
                

            elif process_select.get() == "Canny":

                result = cv.Canny(image_selection, parameter_values[1], parameter_values[0])
                                

            elif process_select.get() == "Laplace":

                im = cv.GaussianBlur(image_selection, (3, 3), 0)
                result = cv.Laplacian(im, cv.CV_8U, ksize=parameter_values[0])
                print(result.dtype)
                result = result.astype('uint8')


            elif process_select.get() == "Aesop":
                
                temp_image = np.copy(image_selection)

                temp_image[temp_image > 0] = 1

                if parameter_values[3] == 0:
                    series_val = True
                else:
                    series_val = False
                
                result = aesop.aesops_Filter(temp_image, kernel_size_start=parameter_values[0], kernel_size_end=parameter_values[1], kernel_step=parameter_values[2], series=series_val, steps=parameter_values[4])

                result[result > 0] = 255

            elif process_select.get() == "Math":
            
                if (math_type.get() == 'Arithmetic') | (math_type.get() == 'Compare'):
                    print(const.get())
                    if const.get() == True:
                        operand2 = math_constant.get()
                    
                    else:
                        operand2 = images[image_names.index(second_select_src.get())]
                        print(operand2.dtype)
                
                

                elif math_operation.get() == '(im)^n' :
                    operand2 = math_constant.get()
                
                if math_operation.get() == 'Add':
                    result = cv.add(image_selection, operand2)
                    print(math_operation.get())
                
                elif math_operation.get() == 'Subtract':
                    result = cv.subtract(image_selection, operand2)
                    print(math_operation.get())

                elif math_operation.get() == 'Multiply':
                    operand2 = operand2.astype('float32') / 255
                    image_selection.astype('float32')
                    result = image_selection * operand2
                    result = result.astype('uint8')
                    #result = cv.multiply(image_selection, operand2)
                    print(math_operation.get())

                elif math_operation.get() == 'Divide':
                    result = cv.divide(image_selection, operand2)
                    print(math_operation.get())

                elif math_operation.get() == 'Max':
                    result = cv.max(image_selection, operand2)
                    print(math_operation.get())

                elif math_operation.get() == 'Min':
                    result = cv.min(image_selection, operand2)
                    print(math_operation.get())

                elif math_operation.get() == '(im)^n':
                    result = image_selection.astype('float32')
                    result = result/255
                    result = cv.pow(result, operand2)
                    result = 255 * result
                    result = result.astype('uint8')
                    print(math_operation.get())

                elif math_operation.get() == 'e^(im)]':
                    result = image_selection.astype('float32')
                    print(result.max())
                    result = result/255
                    print(result.max())
                    result = np.exp(image_selection)
                    result = 255 * result
                    result = result.astype('uint8')                    
                    print(math_operation.get())
                    
                elif math_operation.get() == 'log(im)':
                    result = image_selection.astype('float32')
                    result = result/255                    
                    result = np.log(image_selection)
                    result = 255 * result
                    result = result.astype('uint8')                      
                    print(math_operation.get())

                math_routine()

            elif process_select.get() == "Thresholding":
                print(process_select.get())
                if thresh_select.get() == 'Binary':
                    ret, result = cv.threshold(image_selection, parameter_values[0], parameter_values[1], cv.THRESH_BINARY)

                elif thresh_select.get() == 'Inverse_Binary':
                    ret, result = cv.threshold(image_selection, parameter_values[0], parameter_values[1], cv.THRESH_BINARY_INV)

                elif thresh_select.get() == 'Truncated':
                    ret, result = cv.threshold(image_selection, parameter_values[0], parameter_values[1], cv.THRESH_TRUNC)

                elif thresh_select.get() == 'To_Zero':
                    ret, result = cv.threshold(image_selection, parameter_values[0], parameter_values[1], cv.THRESH_TOZERO)                

                elif thresh_select.get() == 'Inverse_To_Zero':
                    ret, result = cv.threshold(image_selection, parameter_values[0], parameter_values[1], cv.THRESH_TOZERO_INV)

                elif thresh_select.get() == 'Otsu_Bin':
                    ret, result = cv.threshold(image_selection, parameter_values[0], parameter_values[1], cv.THRESH_BINARY+cv.THRESH_OTSU)                

                elif thresh_select.get() == 'Triangle_Bin':
                    ret, result = cv.threshold(image_selection, parameter_values[0], parameter_values[1], cv.THRESH_BINARY+cv.THRESH_TRIANGLE)

                elif thresh_select.get() == 'Adaptive_Thresh_Mean_C':
                    result = cv.adaptiveThreshold(image_selection, parameter_values[0], cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, parameter_values[1], parameter_values[2])
                    
                elif thresh_select.get() == 'Adaptive_Thresh_Gaussian_C':
                    result = cv.adaptiveThreshold(image_selection, parameter_values[0], cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, parameter_values[1], parameter_values[2])                
                
                elif thresh_select.get() == 'Trunc_Hist':
                    result = np.copy(image_selection)
                    histogram = np.histogram(result.ravel(), bins=256, range=[0,256])
                    zeroed = np.where([histogram[0] > 10**(trunc_hist_scale.get())])[1]
                    for j in zeroed:
                        result[result==j]=0
                    parameter_values = [trunc_hist_scale.get()]

            elif process_select.get() == "Normalize":

                if norm_select.get() == 'Norm_Inf':
                    result = cv.normalize(image_selection, result, alpha=parameter_values[0], beta=parameter_values[1], norm_type=cv.NORM_INF, dtype=cv.CV_8U)

                elif norm_select.get() == 'Norm_L1':
                    parameter_values[0] = float(parameter_values[0])/255
                    parameter_values[1] = float(parameter_values[1])/255
                    print(parameter_values)
                    
                    image_selection = (image_selection.astype('float32'))/255
                    print(image_selection.max(), image_selection.min())
                    #result = cv.normalize(image_selection, result, alpha=parameter_values[0], beta=parameter_values[1], norm_type=cv.NORM_L1, dtype=cv.CV_32F)
                    result = cv.normalize(image_selection, result, alpha=1.0, beta=0.0, norm_type=cv.NORM_L1, dtype=cv.CV_32F)
                    result = result / result.max()
                    result = (result * 255).astype('uint8')
                    

                elif norm_select.get() == 'Norm_L2':
                    parameter_values[0] = float(parameter_values[0])/255
                    parameter_values[1] = float(parameter_values[1])/255
                    print(parameter_values)
                    
                    image_selection = (image_selection.astype('float32'))/255
                    print(image_selection.max(), image_selection.min())
                    #result = cv.normalize(image_selection, result, alpha=parameter_values[0], beta=parameter_values[1], norm_type=cv.NORM_L2, dtype=cv.CV_8U)
                    result = cv.normalize(image_selection, result, alpha=1.0, beta=0.0, norm_type=cv.NORM_L2, dtype=cv.CV_32F)
                    result = result / result.max()
                    result = (result * 255).astype('uint8')

                elif norm_select.get() == 'Norm_L2_Square':
                    parameter_values[0] = float(parameter_values[0])/255
                    parameter_values[1] = float(parameter_values[1])/255
                    print(parameter_values)
                    
                    image_selection = (image_selection.astype('float32'))/255
                    print(image_selection.max(), image_selection.min())                    
                    #result = cv.normalize(image_selection, result, alpha=parameter_values[0], beta=parameter_values[1], norm_type=cv.NORM_L2SQR, dtype=cv.CV_8U)
                    result = cv.normalize(image_selection, result, alpha=1.0, beta=0.0, norm_type=cv.NORM_L2SQR, dtype=cv.CV_32F)
                    result = result / result.max()
                    result = (result * 255).astype('uint8')

                elif norm_select.get() == 'Norm_Hamming':
                    result = cv.normalize(image_selection, result, alpha=parameter_values[0], beta=parameter_values[1], norm_type=cv.NORM_HAMMING, dtype=cv.CV_8U)

                elif norm_select.get() == 'Norm_Hamming2':
                    result = cv.normalize(image_selection, result, alpha=parameter_values[0], beta=parameter_values[1], norm_type=cv.NORM_HAMMING2, dtype=cv.CV_8U)

                elif norm_select.get() == 'Norm_Relative':
                    result = cv.normalize(image_selection, result, alpha=parameter_values[0], beta=parameter_values[1], norm_type=cv.NORM_RELATIVE, dtype=cv.CV_8U)

                elif norm_select.get() == 'Norm_Min_Max':
                    result = cv.normalize(image_selection, result, alpha=parameter_values[0], beta=parameter_values[1], norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


            elif process_select.get() == "Miscellaneous":

                if misc_select.get() == 'Invert':
                    result = (255 * np.ones(image_selection.shape, dtype='uint8')) - image_selection

                elif misc_select.get() == 'Resize':
                    if resize_select.get() == 'Reduce':
                        scale = scale_scale.get()
                        scaled_size = (int(image_selection.shape[0] * scale),int(image_selection.shape[1] * scale) )
                        result = resize(image_selection, scaled_size, anti_aliasing=True)
                        result = (result*255).astype('uint8')
                        parameter_values = [scale]

                    elif resize_select.get() == 'Match_Size':
                        result = resize(image_selection, images[image_names.index(second_select_src.get())].shape)
                        result = (result*255).astype('uint8')
                        parameter_values = [image_names.index(second_select_src.get())]
                    




            elif process_select.get() == "Morphology":

                # Set the kernel for morphological operation
                print(morph_kernel_select.get())

                if morph_kernel_select.get() == 'Rectangle':
                    morph_kernel = cv.getStructuringElement(cv.MORPH_RECT,( parameter_values[0], parameter_values[1]))

                elif morph_kernel_select.get() == 'Cross':
                    morph_kernel = cv.getStructuringElement(cv.MORPH_CROSS,( parameter_values[0], parameter_values[1]))

                elif morph_kernel_select.get() == 'Ellipse':
                    morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,( parameter_values[0], parameter_values[1]))

                # Apply the morphological operation

                if morph_select.get() == 'Erode':
                    result = cv.erode(image_selection, morph_kernel, iterations = parameter_values[2])

                elif morph_select.get() == 'Dilate':
                    result = cv.dilate(image_selection, morph_kernel, iterations = parameter_values[2])

                elif morph_select.get() == 'Open':
                    result = cv.morphologyEx( image_selection, cv.MORPH_OPEN, morph_kernel)

                elif morph_select.get() == 'Close':
                    result = cv.morphologyEx( image_selection, cv.MORPH_CLOSE, morph_kernel)

                elif morph_select.get() == 'Morph_Gradient':
                    result = cv.morphologyEx( image_selection, cv.MORPH_GRADIENT, morph_kernel)

                elif morph_select.get() == 'Top_Hat':
                    result = cv.morphologyEx( image_selection, cv.MORPH_TOPHAT, morph_kernel)

                elif morph_select.get() == 'Black_Hat':
                    result = cv.morphologyEx( image_selection, cv.MORPH_BLACKHAT, morph_kernel)

            #print(type(image_selection), parameter_values[0], parameter_values[1])
            print(type(result))
            cv.imshow('Result', result)

            plot_hist(result, 'Result Histogram')

            print(time.time() - t1)
            




########################     Save Mask and Record processing steps for use in batch processing.    #####################


        def apply_result():
            global mask_count
            global image_source
            global result
            global operand2
            global save_mask_win
            global truncate
            global truncate_scale

            apply_button['state'] = 'disabled'

            try:
                if 'normal' == save_mask_win.state():
                    save_mask_button['state'] = 'disabled'
            except:
                save_mask_button['state'] = 'normal'
            save_process_button['state'] = 'normal' 

            #image_names.append(process_select.get()+ '_' + str(mask_count) + '_' + str(image_names.index(image_select_src.get())))
            
            print(image_names)
            images.append(result)

            if process_select.get() in [ "Gabor", "Log_Gabor", "Gauss", "Canny", "Laplace", "Aesop"]:

                image_names.append(process_select.get()+ '_' + str(mask_count) + '_' + str(image_names.index(image_select_src.get())))

                process_summary.append( {   'im_name'           : image_names[-1],
                                            'src_im_ind'        : image_names.index(image_select_src.get()),
                                            'source_im_name'    : image_select_src.get(),
                                            'process'           : process_select.get(),
                                            'parameters'        : parameter_values
                                            })

                if process_select.get() == "Log_Gabor":
                    process_summary[-1]["LG_Normalize"] = LG_normalize.get()
                    process_summary[-1]["truncate"] = truncate.get()
                    if truncate.get() == True:
                        process_summary[-1]["trunc_exp"] = truncate_scale.get()

                
            
            elif process_select.get() == 'Math':
                
                second_term = 'N/A'
                second_term_name = 'N/A'
                constant_term = 'N/A'

                print(math_operation.get())

                if (math_type.get() == 'Arithmetic') | (math_type.get() == 'Compare'):

                    if const.get() == False:

                        second_term = image_names.index(second_select_src.get())
                        second_term_name = second_select_src.get()
                        image_names.append(math_type.get()+ '_' + str(mask_count) + '_' + str(image_names.index(image_select_src.get())) + '_' + str(second_term))

                    else:

                        constant_term = operand2
                        image_names.append(math_type.get()+ '_' + str(mask_count) + '_' + str(image_names.index(image_select_src.get())) + '_c' + str(constant_term))


                elif math_operation.get() == '(im)^n':

                    image_names.append(math_type.get()+ '_' + str(mask_count) + '_' + str(image_names.index(image_select_src.get())) + '_c' + str(math_constant.get))


                elif math_type.get() == 'Logs_and_Exponents':

                    image_names.append(math_type.get()+ '_' + str(mask_count) + '_' + str(image_names.index(image_select_src.get())))


                process_summary.append( {   'im_name'           : image_names[-1],
                                            'src_im_ind'        : image_names.index(image_select_src.get()),
                                            'source_im_name'    : image_select_src.get(),
                                            'src_im_ind2'       : second_term,
                                            'source_im_name2'   : second_term_name,
                                            'process'           : math_op,
                                            'constant_term'     : constant_term
                                            })

            elif process_select.get() == 'Thresholding':

                image_names.append(thresh_select.get()+ '_' + str(mask_count) + '_' + str(image_names.index(image_select_src.get())))

                process_summary.append( {   'im_name'           : image_names[-1],
                                            'src_im_ind'        : image_names.index(image_select_src.get()),
                                            'source_im_name'    : image_select_src.get(),
                                            'process'           : thresh_select.get(),
                                            'parameters'        : parameter_values
                                            })

            elif process_select.get() == 'Normalize':
                
                image_names.append(norm_select.get()+ '_' + str(mask_count) + '_' + str(image_names.index(image_select_src.get())))

                process_summary.append( {   'im_name'           : image_names[-1],
                                            'src_im_ind'        : image_names.index(image_select_src.get()),
                                            'source_im_name'    : image_select_src.get(),
                                            'process'           : norm_select.get(),
                                            'parameters'        : parameter_values
                                            })

            elif process_select.get() == 'Miscellaneous':
                
                image_names.append(misc_select.get()+ '_' + str(mask_count) + '_' + str(image_names.index(image_select_src.get())))

                if misc_select.get() == 'Invert':
                    process_summary.append( {   'im_name'           : image_names[-1],
                                                'src_im_ind'        : image_names.index(image_select_src.get()),
                                                'source_im_name'    : image_select_src.get(),
                                                'process'           : misc_select.get(),
                                                })

                elif misc_select.get() == 'Resize':

                    process_summary.append( {   'im_name'           : image_names[-1],
                                                'src_im_ind'        : image_names.index(image_select_src.get()),
                                                'source_im_name'    : image_select_src.get(),
                                                'process'           : misc_select.get(),
                                                'Resize_Op'         : resize_select.get(),
                                                'parameters'        : parameter_values
                                                })

                    if resize_select.get() == 'Reduce':
                        process_summary[-1]['scale'] = scaled_size
                        print(process_summary[-1])

                    elif resize_select.get() == 'Match_Size':
                        process_summary[-1]['match_im_size'] = second_select_src.get()
                        process_summary[-1]['match_im_size_ind'] = image_names.index(second_select_src.get())
                        print(process_summary[-1])

            elif process_select.get() == "Morphology":

                image_names.append(morph_select.get()+ '_' + str(mask_count) + '_' + str(image_names.index(image_select_src.get())))

                process_summary.append( {   'im_name'           : image_names[-1],
                                            'src_im_ind'        : image_names.index(image_select_src.get()),
                                            'source_im_name'    : image_select_src.get(),
                                            'process'           : morph_select.get(),
                                            'kernel'            : morph_kernel_select.get(),
                                            'parameters'        : parameter_values
                                            })

            for item_number in range(len(process_summary)):
                print('\n\n')
                for dict_item in process_summary[item_number]:
                    
                    print(dict_item, ' : ', process_summary[item_number][dict_item])

            cv.imshow(image_names[-1], images[-1])
            image_source.place_forget()
            image_source = OptionMenu(process_ops_frame, image_select_src, *image_names)
            image_source.place(x=200, y=10)

            mask_count+=1

        
##################################      Save Processed Images       ###########################################################

        def save_mask():

            global save_mask_button
            global save_mask_win

            save_mask_button['state'] = 'disabled'

            save_mask_win = Toplevel()
            save_mask_win.geometry("200x200")
            save_mask_win.title("Save Mask")

            def preview_save():

                image_select = images[image_names.index(save_mask_selection.get())]

                plot_hist(image_select, (save_mask_selection.get()+' Result Histogram'))

                cv.imshow('Saved Mask Preview', image_select)

            def save_image():
                save_image_filename = filedialog.asksaveasfilename(filetypes = [('jpeg image files', '*.jpg')])
                print(save_image_filename)
                cv.imwrite(save_image_filename, images[image_names.index(save_mask_selection.get())])

            def close_all_previews():
                cv.destroyAllWindows()
                plt.close('all')

            def on_save_win_closing():
                save_mask_win.destroy()
                save_mask_button['state'] = 'normal'

            save_mask_win.protocol("WM_DELETE_WINDOW", on_save_win_closing)

            save_mask_selection = StringVar()

            save_image_options = image_names
            save_mask_source = OptionMenu(save_mask_win, save_mask_selection, *save_image_options)
            save_mask_source.place(x=10, y=10)

            Button(save_mask_win, text = 'Preview', command=preview_save).place(x=10, y=50, width = 180, height = 40)
            Button(save_mask_win, text = 'Save Image', command=save_image).place(x=10, y=100, width = 180, height = 40)
            Button(save_mask_win, text = 'Close Image Previews', command=close_all_previews).place(x=10, y=150, width = 180, height = 40)

#####################################      Save Process Summary      ###########################################################

        def save_process():
            
            save_folder = filedialog.askdirectory(initialdir="/Users/anthonyesposito/Pictures", title="Select a Folder")
            print(save_folder)
            now = datetime.now()
            folder_name = save_folder+'/ProcSession_{}{}{}_{}{}/'.format(now.year, now.month, now.day, now.hour, now.minute)
            print(folder_name)
            os.mkdir(folder_name)
            
            process_filename = folder_name + 'ProcessSummary.json'
            with open(process_filename, 'w') as fout:
                json.dump(process_summary , fout)
            
            response =  messagebox.askquestion("Save masks?", "Would you like to save the masks?")
            if response == 'yes':
                mask_folder = folder_name + 'masks/'
                os.mkdir(mask_folder)
                for i in range(len(images)-1):
                    cv.imwrite(mask_folder + image_names[i+1] + '.jpg', images[i+1])


#####################################      Processing Buttons      ###########################################################            

        preview_button = Button(proc_dev_win, text="Preview Result", padx= 10, pady=10, command=preview_result)
        preview_button.place(x= 410, width=100, y = 660, height= 30)
        
        apply_button = Button(proc_dev_win, text="Apply Result", state=DISABLED, padx= 10, pady=10, command=apply_result)
        apply_button.place(x= 410, width=100, y = 700, height= 30)
        
        save_mask_button = Button(proc_dev_win, text="Display/Save", state=DISABLED, padx= 10, pady=10, command=save_mask)
        save_mask_button.place(x= 520, width=100, y = 660, height= 30)

        save_process_button =  Button(proc_dev_win, text="Save Process", state=DISABLED, padx= 10, pady=10, command=save_process)
        save_process_button.place(x= 520, width=100, y = 700, height= 30)

        if len(images) > 1:
            save_process_button['state'] = 'normal'
            


#####################################      Build the process selection frame    ################################################


    process_ops_dict =   {      "Gabor": [
                                            ['ksize', ' - kernel size', 3, 100, 1, 'scale'], 
                                            ['sigma', ' - standard dev.', 1, 5, .1, 'scale'], 
                                            ['theta', ' - orientation angle', 0, 2, .05, 'scale'], 
                                            ['lambda', ' - wavelength', 1, 10, 1, 'scale'], 
                                            ['gamma', ' - aspect ratio', 0.05, 1, .01, 'scale'], 
                                            ['psi', ' - phase offset', 0, 2, .25, 'scale']],

                                "Log_Gabor": [
                                            ['f0', ' - mean frequency', 1, imgG.shape[1]//4, 1, 'scale'],
                                            ['sigmaOnf', ' - standard deviation of filter', 0.005, 1, .005, 'scale'],
                                            ['angle', ' - filter angle', 0., 1.95, .05, 'scale'],
                                            ['thetaSigma', ' - standard deviation of the angle', 0.1, 3, .05, 'scale']],

                                "Gauss": [
                                            ['size', ' - kernel size', 3, 100, 2, 'scale']],
                                
                                "Canny": [
                                            ['max', ' - strong edge threshold', 1, 255, 1, 'scale'],
                                            ['min', ' - weak edge threshold', 0, 254, 1, 'scale']],
                                
                                "Laplace": [
                                            ['size', ' - kernal size', 3, 31, 2, 'scale']],

                                "Aesop": [  ['kernel_size_start', ' - first kernel size', 3, 51, 1, 'scale'], 
                                            ['kernel_size_end', ' - final kernel size', 3, 51, 1, 'scale'], 
                                            ['kernel_step', ' - kernel size change', 1, 51, 1, 'scale'],
                                            ['Series', ' - 0 False, 1 True', 0, 1, 1, 'scale'], 
                                            ['steps', ' - ', 1, 51, 1, 'scale']],

                                "Math": [   ['Arithmetic', 'Compare', 'Logs_and_Exponents'],

                                            {
                                            'Arithmetic': ['Add', 'Subtract', 'Multiply', 'Divide'],
                                            'Compare': ['Max', 'Min'],
                                            'Logs_and_Exponents': ['(im)^n'] #, 'e^(im)]', 'log(im)' ]
                                            }],
                                
                                "Thresholding":
                                        [   [   'Binary', 
                                                'Inverse_Binary', 
                                                'Truncated', 
                                                'To_Zero', 
                                                'Inverse_To_Zero', 
                                                'Adaptive_Thresh_Mean_C', 
                                                'Adaptive_Thresh_Gaussian_C', 
                                                'Otsu_Bin', 
                                                'Triangle_Bin',
                                                'Trunc_Hist'],
                                            
                                            [   ['Threshold', ' - boundary value', 0, 255, 1],
                                                ['Maximum_Value', ' - binary value', 0,255,1],
                                                ['Block_Size', ' - area for threshold averaging', 3,255,2],
                                                ['Constant_Offset', ' - binary offset', 0,255,1]]],

                                "Normalize":
                                        [   [   'Norm_Inf', 
                                                #'Norm_L1', 
                                                #'Norm_L2', 
                                                #'Norm_L2_Square', 
                                                #'Norm_Hamming', 
                                                #'Norm_Hamming2', 
                                                #'Norm_Relative', 
                                                'Norm_Min_Max'],
                                            
                                            [   ['Alpha', ' - lower bound or norm value', 0, 255, 1],
                                                ['beta', ' - upper bound or N/A', 0,255,1]]],
                                
                                "Miscellaneous":
                                        [   [   'Invert',
                                                'Resize']],

                                "Morphology":
                                        [   [   'Erode',
                                                'Dilate',
                                                'Open',
                                                'Close',
                                                'Morph_Gradient',
                                                'Top_Hat',
                                                'Black_Hat'],
                                                
                                            [   'Rectangle',
                                                'Cross',
                                                'Ellipse'],
                                                
                                            [   ['Kernel_Height', ' - Height of the morphological kernel', 3, 300, 1],
                                                ['Kernel_Width', ' - Width of the morphological kernel', 3, 300, 1],
                                                ['Iterations', ' - number of iterations', 1, 100, 1]]]
                            }

    process_frame = LabelFrame(proc_dev_win, text = "Process Selection")
    process_frame.place(x=20, width=370, y=40, height=700) 
    processes = list(process_ops_dict.keys())

    process_select = StringVar()
    process_select.set("N/A")
    
    for i in range(len(processes)):
        Radiobutton( process_frame, text=processes[i], variable=process_select, command=process_options, value=processes[i]).place(relx=.1, rely=(.05 + .8 *( i / (len(processes)-1))))
    
    
    

    


#####################################      Open the file for process development    ################################################

def open_proc_dev_file():

    global img
    global imgG
    global images
    global image_names
    global process_summary
    global mask_count
    

    root.filename = filedialog.askopenfilename(initialdir="/Users/anthonyesposito/Pictures", title="Select a File", filetypes=(('jpg files', '*.jpg'),('JPG files', '*.JPG'), ('png files', '*.png')))
    img = cv.imread(root.filename, cv.IMREAD_COLOR)
    

    cv.imshow('Base_Image', img)
    cv.waitKey(1)

    imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    images = list([imgG])
    image_names = ['Base_Image']
    
    
    mask_count = 1

    process_summary = []
                             

    cv.imshow('Base_Image_Gray', images[0])
    proc_dev_interface()

    def on_process_closing():
        if messagebox.askokcancel("Leave Process Development", "Do you want to leave process development?"):
            if len(images) > 1:
                if messagebox.askyesno("Save Work?", "Would you like to save your progress?"):
                    save_process()
            proc_dev_win.destroy()
            cv.destroyAllWindows()
            dev_im_proc_button["state"] = "normal"
            batch_proc_button["state"] = "normal"
        plt.close('all')

    proc_dev_win.protocol("WM_DELETE_WINDOW", on_process_closing)


######################################       Open the batch processing window    ################################################
def open_batch_proc():

    global batch_proc_win
    global images_save
    global images_save_tk
    global save_list
    global images_out
    global images_out_tk
    global out_list
    global process_summary_filename


    # Import The Summary and Select Process Steps to be saved 
    batch_proc_win = Toplevel()
    batch_proc_win.title("Batch Processing - Image Save Selection")
    batch_proc_win.geometry("500x200")

    Label(batch_proc_win, text="Available Process Steps").place(x=115, y=10, anchor=N)
    Label(batch_proc_win, text="Process Steps To Be Saved").place(x=375, y=10, anchor=N)

    def batch_save_listbox_place():

        global images_save
        global images_save_tk
        global save_list
        global images_out
        global images_out_tk
        global out_list

        
        
        images_save_tk = StringVar(value=images_save)
        save_list_sb = Scrollbar(batch_proc_win)
        save_list_sb.place(x=470,y=40, height=100)
        save_list = Listbox(batch_proc_win, listvariable=images_save_tk, height=6, selectmode='extended', yscrollcommand = save_list_sb.set)
        save_list.place(x = 375, y=40, anchor=N)


        images_out_tk = StringVar(value=images_out)
        out_list_sb = Scrollbar(batch_proc_win)
        out_list_sb.place(x=210,y=40, height=100)
        out_list = Listbox(batch_proc_win, listvariable=images_out_tk, height=6, selectmode='extended', yscrollcommand = out_list_sb.set)
        out_list.place(x = 115, y=40, anchor=N)

        

    process_summary_filename = filedialog.askopenfilename(initialdir="/Users/anthonyesposito/Pictures", title="Select JSON Process Summary File", filetypes=(('json files', '*.json'),('JPG files', '*.JPG'), ('png files', '*.png')))
    proc_summ_json = open(process_summary_filename)
    
    process_summary_dict = json.load(proc_summ_json)
    
    image_names = []

    for i in range(len(process_summary_dict)):
        image_names.append(process_summary_dict[i]['im_name'])

    images_save = []

    images_save.append(image_names[-1])

    images_out = []

    for i in image_names:
        if i not in images_save:
            images_out.append(i)

    batch_save_listbox_place()    
    
    def reset_name_lists():

        global imsave_temp
        global images_save
        global images_save_tk
        global save_list
        global images_out
        global images_out_tk
        global out_list

        images_save = []

        for i in image_names:
            if i in imsave_temp:
                images_save.append(i)        
        
        images_out = []

        for i in image_names:
            if i not in images_save:
                images_out.append(i)



    # to save list
    def to_save_list():

        global imsave_temp
        global images_save
        global images_save_tk
        global save_list
        global images_out
        global images_out_tk
        global out_list
        
        imsave_temp = images_save.copy()

        for i in out_list.curselection():
            imsave_temp.append(images_out[i])

        reset_name_lists()

        save_list.place_forget()
        out_list.place_forget()
        
        batch_save_listbox_place()

    to_save_list_button = Button(batch_proc_win, padx=2, text=" > ", command=to_save_list)
    to_save_list_button.place(x=250, y=40, anchor=N)

    # all to save list

    def all_to_save_list():

        global imsave_temp
        global images_save
        global images_save_tk
        global save_list
        global images_out
        global images_out_tk
        global out_list

        imsave_temp = image_names.copy()

        reset_name_lists()

        save_list.place_forget()
        out_list.place_forget()
        
        batch_save_listbox_place()

    all_to_save_list_button = Button(batch_proc_win, padx=2, text=">>", command=all_to_save_list)
    all_to_save_list_button.place(x=250, y=65, anchor=N)

    # all to out list

    def all_to_out_list():

        global imsave_temp
        global images_save
        global images_save_tk
        global save_list
        global images_out
        global images_out_tk
        global out_list
        
        imsave_temp = []

        for i in save_list.curselection():
            imsave_temp.remove(images_save[i])

        reset_name_lists()

        save_list.place_forget()
        out_list.place_forget()
        
        batch_save_listbox_place()

    all_to_out_list_button = Button(batch_proc_win, padx=2, text="<<", command=all_to_out_list)
    all_to_out_list_button.place(x=250, y=90, anchor=N)

    # to out list

    def to_out_list():

        global imsave_temp
        global images_save
        global images_save_tk
        global save_list
        global images_out
        global images_out_tk
        global out_list
        
        imsave_temp = images_save.copy()

        for i in save_list.curselection():
            imsave_temp.remove(images_save[i])

        reset_name_lists()

        save_list.place_forget()
        out_list.place_forget()
        
        batch_save_listbox_place()

    to_out_list_button = Button(batch_proc_win, padx=2, text=" < ", command=to_out_list)
    to_out_list_button.place(x=250, y=115, anchor=N)

    def image_select_window():

        global batch_proc_win
        global images_save

        batch_proc_win.destroy()

        image_filenames = filedialog.askopenfilenames(initialdir="/Users/anthonyesposito/Pictures", title="Select Images for Batch Processing", filetypes=(('jpg files', '*.jpg'),('JPG files', '*.JPG'), ('png files', '*.png')))

        image_names = []

        for path in image_filenames:
            head, tail = os.path.split(path)
            image_names.append(tail)

        for i in image_names:
            print(i)

        batch_proc_win = Toplevel()
        batch_proc_win.title("Batch Processing - Image Save Selection")
        batch_proc_win.geometry("250x370")

        image_names_tk = StringVar(value=image_names)

        Label(batch_proc_win, text="Images to be Processed").place(x=115, y=10, anchor=N)
        images_list_sb = Scrollbar(batch_proc_win)
        images_list_sb.place(x=210,y=40, height=255)
        images_list = Listbox(batch_proc_win, listvariable=image_names_tk, height=15, selectmode='extended', yscrollcommand = images_list_sb.set)
        images_list.place(x = 115, y=40, anchor=N)

        select_new_set_button = Button(batch_proc_win, text="Select a New Image Set", width=16, command=image_select_window)
        select_new_set_button.place(x=115, y=300, anchor=N)

        def batch_process_images():

            global batch_proc_win
            global process_summary_filename
            global images_save

            batch_proc_win.destroy()

            Stacking.batch_process(process_summary_filename, image_filenames, process_summary_dict, images_save, 'batch')
            
            print('Batch Processing Complete')

        batch_process_button = Button(batch_proc_win, text="Process Images", width=16, command=batch_process_images)
        batch_process_button.place(x=115, y=330, anchor=N)               

    
    image_select_window_button = Button(batch_proc_win, text="Next : Select Images", command=image_select_window)
    image_select_window_button.place(x=375, y=150, anchor=N)


######################################       Open the focus stacking window    ################################################    

def focus_stack_proc():

    global focus_stack_win

    focus_stack_win = Toplevel()
    focus_stack_win.title("Focus Stacking - Process Selection")
    focus_stack_win.geometry("200x100")

    def image_selection_window():
        
        global focus_stack_win

        try:
            focus_stack_win.destroy()
        except:
            pass

        image_filenames = filedialog.askopenfilenames(initialdir="/Users/anthonyesposito/Pictures", title="Select Images for Batch Processing", filetypes=(('jpg files', '*.jpg'),('JPG files', '*.JPG'), ('png files', '*.png')))

        image_names = []

        for path in image_filenames:
            head, tail = os.path.split(path)
            image_names.append(tail)

        for i in image_names:
            print(i)

        focus_stack_win = Toplevel()
        focus_stack_win.title("Batch Processing - Image Save Selection")
        focus_stack_win.geometry("250x370")

        image_names_tk = StringVar(value=image_names)

        Label(focus_stack_win, text="Images to be Processed").place(x=115, y=10, anchor=N)
        images_list_sb = Scrollbar(focus_stack_win)
        images_list_sb.place(x=210,y=40, height=255)
        images_list = Listbox(focus_stack_win, listvariable=image_names_tk, height=15, selectmode='extended', yscrollcommand = images_list_sb.set)
        images_list.place(x = 115, y=40, anchor=N)

        select_new_set_button = Button(focus_stack_win, text="Select a New Image Set", width=16, command=image_selection_window)
        select_new_set_button.place(x=115, y=300, anchor=N)

        def stack_and_save():

            global process_summary_dict
            
            output_filename = filedialog.asksaveasfilename()

            images, masks, histograms = Stacking.batch_process(None, image_filenames, process_summary_dict, None, 'stack')

            order, trans_on = Stacking.image_sort(images, histograms, hist_min=10, hist_max=255)

            Stacking.reg_comb(images, order, trans_on, masks, output_filename)
            
            pass

        focus_stack_button = Button(focus_stack_win, text="Stack Images", width=16, command=stack_and_save)
        focus_stack_button.place(x=115, y=330, anchor=N)    

    def default_process():
        pass
    
    def user_select_process():

        global process_summary_dict

        process_summary_filename = filedialog.askopenfilename(initialdir="/Users/anthonyesposito/Pictures", title="Select JSON Process Summary File", filetypes=(('json files', '*.json'),('JPG files', '*.JPG'), ('png files', '*.png')))
        
        proc_summ_json = open(process_summary_filename)
        
        process_summary_dict = json.load(proc_summ_json)

        image_selection_window()

    dev_im_proc_button = Button(focus_stack_win, padx=20, pady=20, text="Default Process", command=user_select_process)
    dev_im_proc_button.place(relx=.1, relwidth=.8, rely=.1, relheight=.35)

    dev_im_proc_button = Button(focus_stack_win, padx=20, pady=20, text="User Selected", command=user_select_process)
    dev_im_proc_button.place(relx=.1, relwidth=.8, rely=.55, relheight=.35)



######################################       Set Up Main Window    ################################################

dev_im_proc_button = Button(root, padx=20, pady=20, text="Develop Image Process", command=open_proc_dev_file)
dev_im_proc_button.place(relx=.1, relwidth=.8, rely=.25, relheight=.15)

batch_proc_button = Button(root, padx=20, pady=20, text="Batch Processing", command=open_batch_proc)
batch_proc_button.place(relx=.1, relwidth=.8, rely=.45, relheight=.15)

focus_stack_button = Button(root, padx=20, pady=20, text="Focus Stack", command=focus_stack_proc)
focus_stack_button.place(relx=.1, relwidth=.8, rely=.65, relheight=.15)

quit_button = Button(root, padx=20, pady=20, text="Quit", command=root.quit)
quit_button.place(relx=.1, relwidth=.8, rely=.85, relheight=.1)

def on_closing():
    if messagebox.askokcancel("Quit", "Are you sure you would like to quit?"):
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# run the root
root.mainloop()