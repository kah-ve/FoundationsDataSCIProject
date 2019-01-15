# Foundations of Data Science Final Project

### Given a table of log normalized stock data with 9 unknown companies in the same industry and 21 days of data for each (making 9 * 21 = 189 columns) over the past 2500 days. Total table is 2500 rows with 189 columns. Also given the future values of company 4 as the Y to train against. Need to build the best regression and classification models that will give the best predictions for new company 4 data that will be fed in to our trained models. 

## Results: Ended up using a Random Forest with parameters 

    params = {
     'n_estimators': 1400,
     'min_samples_split': 2,
     'min_samples_leaf': 1,
     'max_features': 'auto',
     'max_depth': 100,
     'bootstrap': True}
     
![Model Performances](https://github.com/kah-ve/FoundationsDataSCIProject/blob/master/ModelResults.png)

# [The Python Notebook](https://github.com/kah-ve/FoundationsDataSCIProject/blob/master/Cam_Abdurrahman.ipynb)

## My Attempt

### Regression:

**Attempt 1 Linear:** Tried simple linear, then ridge and lasso. Did not get good results since the lowest MSE was when all the coefficients were 0 due to a large alpha (lambda) value.

**Attempt 1.5 PCA:** This worked the best with the most straightforward approach to the data, which is just taking the data as it existed in the dataframe given and training on the y labels we were also given. (I changed my approach to the data later) Basically the first component of the PCA reduction gave the best MSE. It was a flat-line as in linear but worked slightly better.

**Attempt 2 SVM:** I had the most hopes for SVM before I started, but it completely failed me. I tried different kernels (rbf, linear, poly) but it didn’t help make it better.

**Attempt 3 Random Forests:** This is the model I eventually used for both Regression and Classification. However, in the initial approach to the data, the random forest did not perform well. Changing Approach to Data: I tried three different things, and the third one finally worked. First was separating each company and checking the correlation between the different companies and company 4. I picked top 3 correlated companies and tried to build a model on those. Finally I beat the PCA MSE using a random forest (again SVM didn’t work); then I tried a random selection of companies instead of just the top 3 correlated ones, that didn’t perform better. So I thought I had plateaued, but had one last try: take every company’s data and merge them together. So take each company and append their data on the same columns (while copying the y values respectively) and then train a model on that. My hypothesis was that the companies probably behaved similarly since they were the same industry and we didn’t need to separate them, but just train a model that understands that specific market. This gave me the best result out of many. Here are the Regression MSEs (not including my final tuned random forest model):

![MSE_Results](https://github.com/kah-ve/FoundationsDataSCIProject/blob/master/MSE_results.png)

You can see that the Random Forest on all the data merged together gave the best MSE. Then I ran a RandomizedSearchCV on some different parameters and eventually found a model that gave an MSE ~4.3% smaller (**0.0001637**) than the initial model I had tuned on just the num_estimators and max_depth (using a simple matrix that stored the MSEs of cross validation as I changed those two parameters).

### Classification:

**Attempt 1 Logistic Classifier:** At this point I didn’t bother trying these classifiers on the previous form of the data. I immediately began working with the data all merged together. The logistic classifier gave me an MSE of 0.4830 and I didn’t try to tune it further.

**Attempt 2 SVC:** By this point I had also given up on the SVM giving good results. I tried it many different ways and tuned it many times in the regression section, but to no avail. So I just ran the data through a SVC and got an MSE of 0.4858 (larger than the logistic classifier).

**Attempt 3 Random Forest Classifier:** Again I started with my naïve technique of just running through the num_estimators and max_depth in two for loops and checking the MSE by cross validation. Then I would select the lowest MSE given by num_estimators and max_depth, and try to narrow the range of my for loops to get a more tuned model. However, I discovered something called RandomizedSearchCV at this point (which let me explore ranges that I couldn’t with GridSearchCV) and after running that for nearly ~3 hours I found parameters much different than I initially had. Also instead of just max_depth and num_estimators I was working with other variables such max_features, min_samples_leaf, min_samples_split, etc.

**Quick aside:** I also tried running a RandomizedSearchCV on the regression side of things after, but for some reason it was MUCH slower. Talking about 4 tasks in 16 minutes, when you will be going through 300 tasks. The classification randomized search took 160 minutes for 300 tasks and was reasonable to run. So I dropped the number of data points from the total 22k (when you merge them) and only tried a randomized search regression tree on 1k data points. It gave me some parameters, but compared to the parameters I got from the classification tree, they performed worse in cross validation. Ideally I would’ve liked to be able to run a search in the regression domain and find parameters more specific to it with the full data, but I skipped it since it would’ve taken too long.

**So with my naïve tuning of the Random Forest Classifier I got an MSE of:** 0.4321 which is significantly better than both Logistic and SVC.

**After the randomized search tuning, I got an MSE of:** 0.4078, a ~6% decrease from my best model. 

**Model Saved Format:** I saved the models with pickle since training them takes some time, but saw that they had sizes in the GBs. So I also zipped them using gzip and am loading them in that format. That brings their total size down to 700MB, down from the original 2.2GB.





