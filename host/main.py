"""
ECG Monitor - 医电心电监护上位机
==================================
主入口文件。运行此文件启动上位机软件。

功能：
  - 实时显示 ECG 心电波形
  - 显示心率 (bpm)
  - 显示导联状态（正常/脱落）
  - 串口通信（可配置波特率，8N1）
  - 数据记录导出为 CSV
  - 通信日志查看

协议帧格式（10 字节）：
  [ModuleID(1B)] [DataHead(1B)] [SecondID(1B)] [Data0..5(6B)] [CheckSum(1B)]

用法：
  python main.py

说明：本程序为课程项目原型，不用于临床诊断、监护或治疗决策。
"""

import sys
import os

# Ensure the script directory is in the path (for PyInstaller)
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from main_window import ECGMainWindow


def main():
    app = QApplication(sys.argv)

    # Set default font
    font = QFont("Microsoft YaHei", 11)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)

    # Create and show main window
    window = ECGMainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
