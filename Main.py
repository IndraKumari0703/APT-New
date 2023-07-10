from Sub_functions import main_perf_evaluation_all,main_perf_plot_all
#####################       Main Code           #############################
import PySimpleGUI as sg
VVV=sg.PopupYesNo('Do You want Complete Execution?')
if (VVV == "Yes"):
    t=0
    main_perf_evaluation_all(t)
    main_perf_plot_all()
else:
    main_perf_plot_all()



# confussion_matrix=confusion_matrix(tst_lab, pred_9, labels=[0, 1, 2, 3,4])
# plot_confusion_matrix(cm= confussion_matrix,normalize    = False, target_names = ['Benign', 'Data Exfiltration', 'Establish Foothold', 'Lateral Movement', 'Reconnaissance'],title = "Confusion Matrix")

