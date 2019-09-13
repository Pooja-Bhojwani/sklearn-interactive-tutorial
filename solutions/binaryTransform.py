from sklearn.pipeline import Pipeline, FeatureUnion

impute_pipeline = Pipeline([
    ('selector', ColumnSelector(impute_columns)),
    ('imputer', SimpleImputer(strategy="median")),
    ])
bin_pipeline = Pipeline([
    ('selector', ColumnSelector(bin_column)),
    ('Binning', BinarizeTransformer(threshold=5)),
    ])
label_encode = Pipeline([
    ('selector', ColumnSelector(label_encode_column)),
    ('LabelEncoder', OrdinalEncoder()),
    ])
one_hot_encode = Pipeline([
    ('selector', ColumnSelector(one_hot_encode_column)),
    ('LabelEncoder', OneHotEncoder()),
    ])
numeric_pipeline = Pipeline([
    ('selector', ColumnSelector(numeric_columns)),
    ('imputer', SimpleImputer(strategy="median")),
    ('Scaler', StandardScaler()),
    ])


full_pipeline = FeatureUnion(transformer_list=[
    ("numeric_pipeline", numeric_pipeline),
    ("bin_pipeline", bin_pipeline),
    ("label_encode", label_encode),
    ("one_hot_encode", one_hot_encode),
    ])

finalpipeline = (make_pipeline(full_pipeline, RandomForestRegressor(random_state=1, 
                                                                          n_jobs=-1, 
                                                                        n_estimators=100)))
# Fitting the pipeline
finalpipeline.fit(x_train, y_train)
