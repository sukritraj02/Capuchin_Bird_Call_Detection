import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def export_results_to_csv(predictions, file_name='results.csv'):
    df = pd.DataFrame(predictions, columns=['Prediction'])
    df.to_csv(file_name, index=False)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
