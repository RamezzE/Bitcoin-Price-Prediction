import pandas as pd

def walk_forward_val(model, X_train, X_test, y_train, y_test):
    history_X = X_train.copy() 
    history_Y = y_train.copy()  
    predictions = []  

    test = X_test.copy()  
    test_y = y_test.copy()

    for i in range(len(test)):
        
        # if i % 10 == 0:
        #     print(f"Iteration {i}/{test.shape[0]}")
            
        model.fit(history_X, history_Y)

        y_pred = model.predict(test.iloc[i:i+1]) 

        predictions.append(y_pred)

        next_observation = test.iloc[i:i+1]
        next_target = test_y.iloc[i]
        
        history_X = pd.concat([history_X, next_observation], ignore_index=True)
        history_Y = pd.concat([history_Y, pd.Series(next_target)], ignore_index=True)

    return predictions
