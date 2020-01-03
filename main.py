import sys
import view

from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)

    appView = view.View()
    appView.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
