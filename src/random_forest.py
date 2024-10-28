import time

import graphviz
import pandas as pd
import plotly.express as px
import sklearn.ensemble
import sklearn.metrics
import sklearn.tree
from tabulate import tabulate


# Generating and saving the Model as a PNG.
def store_to_png(model, train_X, file_name):
    dot_data = sklearn.tree.export_graphviz(model, out_file=None, feature_names=train_X.columns, filled=True)
    graph = graphviz.Source(dot_data)
    png_bytes = graph.pipe(format='png')
    with open(file_name, 'wb') as f:
        f.write(png_bytes)


# computes how much each feature contributes in the given model
def plot_feature_importance(model, names, threshold=None):
    feature_importance_df = pd.DataFrame.from_dict({'feature_importance': model.feature_importances_,
                                                    'feature': names}) \
        .set_index('feature').sort_values('feature_importance', ascending=False)

    if threshold is not None:
        feature_importance_df = feature_importance_df[feature_importance_df.feature_importance > threshold]

    fig = px.bar(
        feature_importance_df,
        text_auto='.2f',
        labels={'value': 'feature importance'},
        title='Feature importances'
    )

    fig.update_layout(showlegend=False)
    fig.show()


# Reading input data from CSV files, merging them together
red_df = pd.read_csv('../wine_quality/winequality-red.csv', sep=';')
red_df['type'] = 'red'
white_df = pd.read_csv('../wine_quality/winequality-white.csv', sep=';')
white_df['type'] = 'white'
df = pd.concat([red_df, white_df])

# convert the type variable, transform red => 0, white => 1
categories = {}
cat_columns = ['type']
for p in cat_columns:
    df[p] = pd.Categorical(df[p])

    categories[p] = df[p].cat.categories

df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
print(categories)
print(tabulate(df, headers='keys', tablefmt='psql'))

# Split between training set and validation set
train_df, val_df = sklearn.model_selection.train_test_split(df, test_size=0.2)

# From each set, pull out the quality column, as it is the predicted feature
train_X, train_y = train_df.drop(['quality'], axis=1), train_df.quality
val_X, val_y = val_df.drop(['quality'], axis=1), val_df.quality

# print(train_X.shape, val_X.shape)

start = time.time()
model = sklearn.tree.DecisionTreeRegressor(max_depth=3)
model.fit(train_X, train_y)
end = time.time()
print(f'Random Forest training in {end - start} seconds')

decision_tree_error = sklearn.metrics.mean_absolute_error(model.predict(val_X), val_y)
print(f'Decision Tree — mean absolute error: {decision_tree_error}')

store_to_png(model, train_X, "decision_tree.png")

start_forest = time.time()
random_forest_model = sklearn.ensemble.RandomForestRegressor(100, min_samples_leaf=100)
random_forest_model.fit(train_X, train_y)
end_forest = time.time()
print(f'Random Forest training in {end_forest - start_forest} seconds')

forest_error = sklearn.metrics.mean_absolute_error(random_forest_model.predict(val_X), val_y)
print(f'Random Forest — mean absolute error: {forest_error}')

plot_feature_importance(random_forest_model, train_X.columns)
