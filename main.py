import tkinter

import cv2
from license_vision import license_filter
# from vehicle import vehicle_filter
# from color import color_filter

from tkinter import *
from tkinter import filedialog
import tkinter.messagebox as mbox
import os
import shutil
import openpyxl

class adress:
    def __init__(self):
        self.adress = 0

    def get_adress(self):
        return self.adress
    def set_adress(self, adress):
        self.adress = adress

A = adress()

excel_ext = r"*.xlsx *.xls *.csv"
img_ext = r"*.jpg *.png *.jpeg *.PNG *.JPG"
def file_find():
    file = filedialog.askopenfilenames(filetypes=( ("img file", img_ext), ("Excel file", excel_ext), ("all file", "*.*")), initialdir=r"C:\Users")
    # print(file)
    # global test_image
    en_filepath.delete(0,END)
    en_filepath.insert(END,file[0])


def file_upload():
    if len(en_filepath.get()) ==0:
        mbox.showinfo("warning", "select file, please")
        return
    else:
        file_name = os.path.abspath(en_filepath.get())
        # dest_path = os.path.join("D:\\",file_name)
        # shutil.copy(en_filepath.get(),dest_path)
        A.set_adress(file_name)
        en_filepath.delete(0,END)
        app.quit()
        return

if __name__ == "__main__":
    # test_img = path #img 입력

    app = Tk()
    en_filepath = Entry(app, width = 100)
    en_filepath.pack(fill="x", padx=1,pady=1)

    fr_bt = Frame(app)
    fr_bt.pack(fill="x", padx = 1, pady = 1)

    bt_upload = Button(fr_bt, text="Upload",width = 10, command = file_upload)
    bt_upload.pack(side="right",padx=1,pady=1)
    bt_find = Button(fr_bt,text = "Find", width =10, command= file_find)
    bt_find.pack(side="right",padx=1,pady =1)

    app.title('license_plate_recognition')
    app.mainloop() #test image 불러오기



    test_image = A.get_adress()

    license_img = license_filter(test_image)
    # car = vehicle_filter(test_image)
    # color = color_filter(test_image)

    # license_img = "123가1234"

    car = "HYUNDAI"
    color = "RED"

    xl_name = "VEHICLE_REGISTER.xlsx"
    book = openpyxl.load_workbook(xl_name)

    sheet = book.worksheets[0]

    def findExistString(searchString,text):
        if searchString in text : return True
        else: return False

    def findNonExistString(searchString, text):
        if searchString in text:return True
        else: return False

    register = False

    for row in sheet.rows:

        if findExistString(license_img,row[1].value):
            if(row[0].value == car and row[2].value == color):
                register = True
            else: register = False

    display = Tk()
    image = tkinter.PhotoImage(file = test_image,master=display)
    display.title("license reading")
    display.geometry("840x800")
    display.resizable(width=True,height=True)

    img = Label(display, image = image)
    label1 = Label(display, text = license_img)
    label2 = Label(display, text = car)
    label3 = Label(display, text = color)
    if register:label4 = Label(display, text = "this vehicle is registered")
    else: label4 = Label(display, text = "this vehicle is not registered")

    img.pack()
    label1.pack()
    label2.pack()
    label3.pack()
    label4.pack()


    display.mainloop() #test image 불러오기



    
    # license_num = license_filter(license_img)

