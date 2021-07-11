# sigir2021challenge

This repo describe the **5th place (according to leaderboard of next item prediction task) solution** to the [Coveo Data Challenge 2021 SIGIR Workshop on eCommerce](https://sigir-ecom.github.io/data-task.html)

### Notebooks
The data challenge has two phases with different testing data
`Stage1_MLP & Stage2_LambdaRank` are for phase 1
`Phase2_Stage1_MLP & Phase2_Stage2_LambdaRank` are for phase 2

They are essentially the same apart from generate submission for different testing dataset in different phase.

### Data processing 

1. product_sku_hash & hased_url which appeared only in testing set or appeared only once in training set are grouped together (treated as one single sku / url)
2. Randomly split all sessions into training set (80%) and validation set (20%)
3. For any single session, it may be transformed into multiple training records according to following logic, e.g. 
    * a session with events `['pageview1', 'sku_A', 'pageview2', 'sku_b','sku_c']` will be transformed into 3 valid training records:
    1. `['pageview1']` with next interacted sku = `sku_A`
    2. `['pageview1', 'sku_A', 'pageview2']` with next interacted sku = `'sku_B'`
    3. `['pageview1', 'sku_A', 'pageview2', 'sku_B']` with next interacted sku = `'sku_C'`
     
### Stage 1 Model (MLP)

For this stage, a neural network based on sku & url embeddings is trained, this is inspired by the [winning approach of 2021 WSDM data challenge](http://ceur-ws.org/Vol-2855/challenge_short_2.pdf)

1. For each records, take the last 5 visited **sku & url** as features (sku_lag_1, sku_lag_2, ... and sku_lag_1, ...)
2. During training, more recent records have higher weights on gradients (sessions in Jan have weights of 1 while sessions in Apr have weights of 4)  
3. Both sku and url embeddings are of dimensions 312  
4. All last 5 visited sku embeddings & last 5 visited url embeddings are concatenated as inputs to the model
5. 3 relu hidden layers, first two are of dimensions 1024, and the last has dimension 312
6. Apply dot product between the last hidden layer and all the sku embeddings
7. Softmax cross entropy loss and adam optimizer is used

### Stage 2 Model (LambdaRank)

This stage takes the top 20 skus with highest probability from stage 1 as input and a [lambdarank](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf) model is trained to rank those 20 skus for each session.

1. each training record from stage 1 is treated as a query, and the top 20 predicted skus as documents
    * so for each query, there are at most 1 positive sample and 19 negative sample
    * for queries where there is no positive sample (which means the next interacted sku is not within top 20 predicted skus from stage 1), they will be excluded in stage 2 
2. following features are used for each query  
    * cosine similarity between the candidate sku (documents) and the sku_lag_1 to 5 of that query
    * cosine similarity between the description embeddings of candidate sku and that of sku_lag_1 to 5
    * cosine similarity between the image embeddings of candidate sku and that of sku_lag_1 to 5
    * whether candidate sku and sku_lag_X are of the same category
    * whether candidate sku and sku_lag_X are of the same price
3. lightgbm with lambdarank objective is used

### Leaderboard metrics

| Model      | Leaderboards (Stage 1) Next Item Prediction |
| ----------- | ----------- |
| MLP (sku & url embedding dimension = 32) | 0.1447 |
| MLP (sku & url embedding dimension = 312)| 0.1540 |
| MLP (include queries from testing data in training) | 0.1996 |
| MLP + LambdaRank | 0.2151 |




