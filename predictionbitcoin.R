#get real timedata using Python
#####python code####
#bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
#bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
#bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0
#bitcoin_market_info['Volume']=bitcoin_market_info['Volume'].astype('int64')
#bitcoin_market_info.to_csv("/Users/xuetan/Desktop/Ivey_Business Analytics/Programming/Rworkfile/bitcoin.csv")

bitcoin<-read.csv("bitcoin.csv")
#train / test
train<-bitcoin[11:1840,]
test<-bitcoin[1:10,]
#import libraries
library("car")
library("caret")
library("forecast")
library("keras")
library("MCMCpack")
library("smooth")
library("tensorflow")
library("tseries")
library("TTR")
library("bsts")

#convert dataframe into time series data
Train<-xts(train[,-1],order.by = as.POSIXct(train$Date))
tsr<-ts(Train[,4],frequency = 365.25,start = c(2013,4,27))
plot(Train$Close,type='l',lwd = 1.5,ylim=c(0,25000),main="Bitcoin Closing Price")
#check for trends and seasonality
dects<-decompose(tsr)#obtain trends and seasonality
plot(dects)
#########################################################
#ARIMA
tsdf<-diff(Train[,4],lag=2)
tsdf<-tsdf[!is.na(tsdf)]
#Augmented Dickey-Fuller Test
adf.test(tsdf)
plot(tsdf, type = 1, ylim = c(-2000, 2000))
#ACF and PACF plots
acf(tsdf)
pacf(tsdf)
gege<-arima(Train[,4],order=c(2,2,1))
# h: number of period of forecasting 
pred<-predict(gege,n.ahead=10)
accuracy(pred$pred,test$Close)
gegef<-as.data.frame(forecast(gege,h=10))
gegefct<-cbind(test,gegef[,1])
plot(forecast(gege,h=10),ylim=c(0,20000))

ggplot() + geom_line(data = gegefct, aes(Date, gegefct[,2],color = "blue")) + geom_line(data = gegefct, aes(Date, gegefct[,3], color = "Dark Red"))

################################################
#Bayesian Regression
#adding linear trend to model
ss<-AddLocalLinearTrend(list(),Train[,4])
#adding seasonal trend to model
ss<-AddSeasonal(ss,Train[,4],nseasons = 365.25)
model1<-bsts(Train[,4],state.specification = ss,niter = 1000)
plot(model1,ylim=c(0,10000)) 
pred1<-predict(model1,horizon=10)
plot(pred1,plot.original=50,ylim=c(0,20000))
plot(pred1,plot.original=1840,ylim=c(0,20000))
accuracy(pred1$mean,test$Close)

#################################################
#svm
library(e1071)
svm <-svm(Close~.,data=train,cost=100,gamma=1)
summary(svm)
svm.pred<-predict(svm,test)
svm.pred
accuracy(svm.pred,test$Close)





