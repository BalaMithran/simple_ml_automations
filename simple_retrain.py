import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
def score()-> int:
    dataset = pd.read_csv('Salary_Data.csv')
    

    X = dataset.iloc[:,:-1].values  #independent variable array
    y = dataset.iloc[:,1].values  #dependent variable vector

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train,y_train) #actually produces the linear eqn for the data

    # use pre existing pickle
    model = pickle.load(open('model.pkl','rb'))

    y_pred = model.predict(X_test) 
    
    return (model.score(X_test, y_test))


def createnewpicklemodel():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle
    dataset = pd.read_csv('Salary_Data.csv')
    

    X = dataset.iloc[:,:-1].values  #independent variable array
    y = dataset.iloc[:,1].values  #dependent variable vector

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train,y_train) #actually produces the linear eqn for the data
    # create pickle
    pickle.dump(regressor, open('model.pkl','wb'))

    # use pickle
    model = pickle.load(open('model.pkl','rb'))

    y_pred = model.predict(X_test) 
    

    # Explained variance score: 1 is perfect prediction
   # print('Variance score: %.2f' % regressor.score(X_test, y_test))
    return (regressor.score(X_test, y_test))



score = score()
if score < .8 :
    print("The old model was not accurate")
    print("Retraining model using updated data")
    
    new_score = createnewpicklemodel()
    print("Retrain complete")
    print("new accuracy is ", new_score)
else:
    print("The old model is still accurate enough")
    print("accuracy is " , score)
    