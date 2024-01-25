from PyQt5 import QtWidgets
import keras
from controller import HomePage

stress = keras.models.load_model('../mode35new_nohappydata.h5')
model = keras.models.load_model('../model_finetune.h5')


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = HomePage(model=model,stress=stress)
    window.show()
    sys.exit(app.exec_())