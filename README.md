# Wifi_location_store
With the rapid popularization of Internet mobile payment, we enjoy more and more conveniences in living. 
Such as when you walk into a restaurant in the mall, the phone will automatically pop up the restaurant coupons; 
when you walk into the mall clothing store, the phone can automatically recommend this shop your favorite clothes; 
passing a mall jewelry When the store, the phone can automatically prompt you for a long time a diamond ring already in stock; 
leave the mall parking, the phone with your permission can automatically pay the car fare. 
These enjoyable services you enjoy are inseparable from behind the big data mining and machine learning support. 
At the right time, the right place to give users the most effective service, is the major new battlefield of intelligent 
development of Internet companies.        

The goal of this competition is to locate the shop where the user is currently 
located. In real life, when a user opens a mobile phone in a shopping mall environment, there are some challenges such as 
inaccurate positioning signals, incomplete environmental information, missing shop information, and too close space in different 
shops. Therefore, how to accurately determine that the shop where the user is located is a problem.

mission details:
Participants will need to conduct data mining and feature creation for each of the August 2017 data stores, users, WIFI, etc., 
and create their own negative samples in the training data for proper machine learning training. 
In our September 2017 data, we use your algorithm or model to determine exactly where the store is located based on 
the user's location and WIFI.

Prediction results with accuracy as the main evaluation criteria.

1. Data source and background

This question provides two kinds of data

(1) store information store information, the training and evaluation are unified.

(2) real users in these shopping malls for a period of time to store transaction data, training and evaluation will 
use different time periods.

 Note: In order to protect the privacy of users and merchants, all data are processed anonymously. 
 At the same time, necessary desensitization measures such as partial sampling and filtering are done. 
 Some of the fields for some of the data may be NULL, so do it yourself.

 

2. Training data

* Shop and mall information form

Field　
shop_id　
Store ID
category_id
Store Type ID　：A total of 40 kinds of types, has been desensitization
longitude
latitude
price
mall_id

