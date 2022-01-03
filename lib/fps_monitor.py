
#import time
import multiprocessing
import sys
import PyQt5.QtGui as QtGui 
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout
 
class Widget_monitor(QMainWindow):
    
    def __init__(self, list_name) :
        super().__init__()
        self.wgt_center = QWidget()
        self.vhl_main = QHBoxLayout()
        self.vbl_info = QVBoxLayout()
        self.fl_info = QFormLayout()
        
        self.lb_frame = QLabel()

        self.list_item = list_name
        self.len_info = len(self.list_item)
        self.list_lb_title = []
        self.list_lb_value = []
        for i in range(self.len_info) :
            lb_title = QLabel(self.list_item[i])
            lb_value = QLabel('0')

            self.list_lb_title.append(lb_title)
            self.list_lb_value.append(lb_value)

        self.initUI()
        return
        
        
    def initUI(self) :
        self.setCentralWidget(self.wgt_center)
        self.wgt_center.setLayout(self.vhl_main)

        self.vhl_main.addWidget(self.lb_frame)
        self.vhl_main.addLayout(self.vbl_info)

        self.vbl_info.addLayout(self.fl_info)

        for i in range(self.len_info) :
            self.fl_info.addRow(self.list_lb_title[i], self.list_lb_value[i])


        
        #self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Monitor')    
        self.show()
        return

    def update_fps(self, list_fps) :
        for i in range(self.len_info) :
            self.list_lb_value[i].setText(str(list_fps[i]))
        return

    def update_frame(self, img) :
        '''
        img is 'opencv' type which can imshow
        '''
        qimg_frame = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.lb_frame.setPixmap(QtGui.QPixmap.fromImage(qimg_frame))
        return

        
        


 