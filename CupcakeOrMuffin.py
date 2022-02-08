#packages for data analysis
import matplotlib
import numpy as np
import pandas as pd
from sklearn import svm

#packages for data visulaization
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
#for jupyter notebook include
#%matplotlib inline

#setting type as integers
dataset = pd.read_csv("Cupcakes vs Muffins.csv")
#type = {"Muffin": 0, "Cupcake": 1}
#dataset.Type = [type[items]for items in dataset.Type]

sns.lmplot("Flour","Sugar", data = dataset, hue = "Type",palette ="Set1", fit_reg = False, scatter_kws = {"s":70})

dataset_features = dataset.columns.values[1:].tolist()
symptoms = dataset[["Flour","Milk","Sugar","Butter","Egg","Baking_Powder","Vanilla"]].values
type_label = np.where(dataset["Type"]=="Muffin", 0,1)

#fitting model
model = svm.SVC(kernel="linear")
print(model.fit(symptoms,type_label))
#generating separate hyperplanes
w = model.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(30,60)
yy = a*xx - (model.intercept_[0])/w[1]

#plotting the parallwls to the separating hyperplane that pass through the support vector
b = model.support_vectors_[0]
yy_down = a*xx + (b[1]-a*b[0])
b = model.support_vectors_[-1]
yy_up = a*xx + (b[1]-a*b[0])


sns.lmplot("Flour","Sugar", data = dataset, hue = "Type",palette ="Set1", fit_reg = False, scatter_kws = {"s":70})
plt.plot(xx,yy, linewidth=2, color="black")
plt.plot(xx,yy_down,"k--")
plt.plot(xx,yy_up,"k--")
plt.show()

#predicting function
def predicting(Flour,Milk,Sugar,Butter,Egg,Baking_Powder,Vanilla):
    if (model.predict([[Flour,Milk,Sugar,Butter,Egg,Baking_Powder,Vanilla]])) == 0:
        print("Muffin")
    else: print("Cupcake")

predicting()







