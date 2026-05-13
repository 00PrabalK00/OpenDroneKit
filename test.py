import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtCore import QFile, QIODevice

app = QApplication(sys.argv)
file = QFile(':/qtwebchannel/qwebchannel.js')
res = file.open(QIODevice.OpenModeFlag.ReadOnly)
with open('test_out.txt', 'w') as f:
    f.write(f'qwebchannel.js exists: {res}\n')
