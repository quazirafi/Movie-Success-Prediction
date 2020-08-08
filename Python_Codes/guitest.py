from Tkinter import *
def printit():
	n = enterstr.get()
	print('this is the text ' + n)
w = Tk()
w.geometry('300x400+700+150')
enterstr = StringVar()
txttable = Entry(w,textvariable = enterstr).grid(row=5,column=6)
btn1 =  Button(w,text='click here',command=printit).grid(row=5,column=6)
w.mainloop()