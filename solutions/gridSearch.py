from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Number of trees in random forest
n_estimators = [100, 500, 750, 1000]
# Number of features to consider at every split
max_features = [3, 4, 5]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 110, num = 4)]
max_depth.append(None)

# Create the random grid
random_grid = {'randomforestregressor__n_estimators': n_estimators,
               'randomforestregressor__max_features': max_features,
               'randomforestregressor__max_depth': max_depth,
              }

print(random_grid)

print("Grid search")
print('\n')

scoring = 'neg_mean_absolute_error'
clf = GridSearchCV(finalpipeline, random_grid, n_jobs=-1, verbose=True, scoring=scoring)
clf.fit(x_train, y_train)

clf_preds = clf.predict(x_test)
clf_preds = pd.Series(clf_preds)
clf_preds = clf_preds.rename("Grid Search Predicted values")
