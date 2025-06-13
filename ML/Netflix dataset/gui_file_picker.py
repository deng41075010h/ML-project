from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()  # 不要跳出主視窗
file_path = askopenfilename()  # 會跳出檔案選取視窗
print(file_path)
