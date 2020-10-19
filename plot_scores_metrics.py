# coding=utf-8 



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

def plot_graph(xvalues, metrics_list,labels,title, legend):
  plt.figure(figsize=(10,8))
  for index,score in enumerate(metrics_list):
        plt.plot(xvalues,score,label=labels[index])
  plt.title(title, fontsize=20)
  plt.legend()
  plt.ylabel("Score")
  plt.xlabel("bit size")
  plt.legend(legend)
  plt.show()


xvalues = ['32', 'q8', 'q7', 'q6', 'q5', 'q4', 'q3']
labels = ['32', 'q8', 'q7', 'q6', 'q5', 'q4', 'q3']

y_fc_inception = [2.6960975, 2.6958268, 2.6776688, 2.682091, 2.6515918, 2.3029766, 1.0173199]
y_cnn_inception = [2.589871, 2.5740933, 2.5407314, 2.5407715, 2.439186, 1.6736695, 1.3600159]

y_fc_precision = [0.7511, 0.7464, 0.7417, 0.7208, 0.6308, 0.4077, 0.0001]
y_fc_recall = [0.6574, 0.6591, 0.6525, 0.6387, 0.563, 0.2885, 0.6631]
y_fc_realism_precision = [1.2095387, 1.202072, 1.2008115, 1.1854748, 1.1314873, 0.96990305, 0.3337342]
y_fc_realism_recall = [1.1447951, 1.1439034, 1.1396828, 1.128987, 1.0821891, 0.88816726, 1.2929996]

y_cnn_precision = [0.4322, 0.4167, 0.3968, 0.365, 0.2146, 0.0415, 0]
y_cnn_recall = [0.1815, 0.1743, 0.1376, 0.1594, 0.0944, 0.0146, 0]
y_cnn_realism_precision = [0.9931411, 0.9796658, 0.96916145, 0.95076126, 0.83588016, 0.60187215, 0.3098696]
y_cnn_realism_recall = [0.8021377, 0.7841936, 0.74731475, 0.7682992, 0.6979954, 0.4938864, 0.17718765]


plot_graph(xvalues, [y_fc_inception, y_cnn_inception], labels, "Inception metric plot", ["FC", "CNN"])

plot_graph(xvalues, [y_fc_precision, y_fc_recall], labels, "Precision and Recall metric plot for FC", ["Precision", "Recall"])
plot_graph(xvalues, [y_cnn_precision, y_cnn_recall], labels, "Precision and Recall metric plot for CNN", ["Precision", "Recall"])

