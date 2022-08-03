from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceTransformer
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data
class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])], remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
        ('linear_model', LinearRegression())])
        
        return pipe

    def run(self):
        """set and train the pipeline"""
        # self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.15,random_state=42)
        self.pipeline=Trainer.set_pipeline(self)
        self.pipeline.fit(self.X, self.y)
        
        
    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    #print('TODO')
    df = get_data()
    df=clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    train=Trainer(X,y)
    train.set_pipeline()
    train.run()
    print('rmse= ',train.evaluate(X_val,y_val))

