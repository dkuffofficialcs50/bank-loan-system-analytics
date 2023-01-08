#Preparing Dataset
credit14 <- read.csv("C:/Users/dkuffofficial/credit14.csv")
library(ISLR)
library(ggplot2)
library(rcompanion)
library(rpart)
library(rpart.plot)
library(aod)
library(MASS)
library(class)
attach(credit14)

Creditability <- factor(Creditability)
Account.Balance<- factor(Account.Balance)
Payment.Status.of.Previous.Credit <- factor(Payment.Status.of.Previous.Credit)
Purpose <- factor(Purpose)
Value.Savings.Stocks <- factor(Value.Savings.Stocks)
Length.of.current.employment <- factor(Length.of.current.employment)
Sex...Marital.Status <- factor(Sex...Marital.Status)
Guarantors <- factor(Guarantors)
Most.valuable.available.asset <- factor(Most.valuable.available.asset)
Concurrent.Credits <- factor(Concurrent.Credits)
Type.of.apartment <- factor(Type.of.apartment)
Occupation <- factor(Occupation)
Telephone <- factor(Telephone)
Foreign.Worker <- factor(Foreign.Worker)


##Method 1 (Decision Tree)

reg.tree <- rpart(Creditability ~ Account.Balance+Duration.of.Credit..month.+Payment.Status.of.Previous.Credit+Purpose+Credit.Amount+Value.Savings.Stocks+Length.of.current.employment+Instalment.per.cent+Sex...Marital.Status+Guarantors+Duration.in.Current.address+Most.valuable.available.asset+Age..years.+Concurrent.Credits+Type.of.apartment+No.of.Credits.at.this.Bank+Occupation+No.of.dependents+Telephone+Foreign.Worker, data = credit14)
rpart.plot(reg.tree, type = 2)

detach(credit14)

##Method 2  (Logistic Regression for Credit 14 data)
View(credit14)
xtabs(~Creditability + Account.Balance, data = credit14) # table of categorical outcome (i.e.Creditability) and variable (i.e.Account Balance)
credit14$Account.Balance <- factor(credit14$Account.Balance)
str(credit14)
logit <- glm(Creditability ~ Duration.of.Credit..month. + Credit.Amount + Account.Balance, data = credit14, family = "binomial")
summary(logit)
confint(logit) # CIs using profiled log-likelihood
confint.default(logit) # CIs using standard errors

## Wald Test (for the overall effect of Account balance)

wald.test(b = coef(logit), Sigma = vcov(logit), Terms = 4:6)

## Wald Test (for the different levels of ACCOUNT Balance, e.g. Account balance 2 and 3)
level<- cbind(0, 0, 0, 1, -1, 0) # create a vector to contrast 
wald.test(b = coef(logit), Sigma = vcov(logit), L = level)
level1<- cbind(0, 0, 0, 0, 1, -1)
wald.test(b = coef(logit), Sigma = vcov(logit), L = level1)

## Odds Ratio
exp(coef(logit)) # OR only
exp(cbind(OR = coef(logit), confint(logit))) # OR and 95% CI


## Predicted Probabilities
# 1 Calculating pp of Creditability at each Account balance
credit14.n1 <- with(credit14, data.frame(Duration.of.Credit..month. = mean(Duration.of.Credit..month.), Credit.Amount = mean(Credit.Amount), Account.Balance = factor(1:4)))
credit14.n1
credit14.n1$AccountbalanceP <- predict(logit, newdata = credit14.n1, type = "response")
credit14.n1

# 2 Calculating pp of Creditability varying the value of Duration of Credit month at each Account Balance
credit14.n2 <- with(credit14, data.frame(Duration.of.Credit..month. = rep(seq(from = 4, to = 72, length.out =225),4), Credit.Amount = mean(Credit.Amount), Account.Balance = factor(rep(1:4, each = 225)))) # 72 Duration of credit month values 
View(credit14.n2)

# 3 Add log-odds, and probability
credit14.n3 <- cbind(credit14.n2, predict(logit, newdata = credit14.n2, type = "link", se = TRUE)) # including standard error for CI later
credit14.n3 <- within(credit14.n3, {
  PredictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})
View(credit14.n3)

## Plot

ggplot(credit14.n3, aes(x = Duration.of.Credit..month., y = PredictedProb)) + 
  geom_ribbon(aes(ymin = LL,ymax = UL, fill = Account.Balance), alpha = 0.2) + geom_line(aes(colour = Account.Balance), size = 1)

## Likelihood Ratio Test
summary(logit)
logLik(logit) # see the model's log-likelihood
with(logit, null.deviance - deviance) # find the difference in deviance
with(logit, df.null - df.residual) # The df for the difference between the two models = the number of predictor variables
with(logit, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE)) # obtain p-value



#Method 3 (Discriminant Analysis)

credit14.lda<-lda(Creditability~Duration.of.Credit..month.+Credit.Amount+Instalment.per.cent+Duration.in.Current.address+Age..years.+No.of.Credits.at.this.Bank+No.of.dependents, data=credit14, prior=c(1,1)/2)
credit14.lda
 
#Create summary of prediction capabilities
credit14.lda.predict<-predict(credit14.lda)
credit14.complete<-credit14[complete.cases(credit14),] #remove NA rows
table(credit14.complete$Creditability, data=credit14.lda.predict$class)
.table<-table(credit14.complete$Creditability, data=credit14.lda.predict$class)
addmargins(.table)
round(addmargins(prop.table(.table,1)*100,2),2) #express table entries as fraction Of marginal table
# prop.table(.table,1) 
ncorrect<-sum(diag(.table))
# diag(.table) 
ntotal<-sum(.table)
cat(ncorrect," correctly allocated out of ",ntotal," (",100*ncorrect/ntotal,"%)","\n")

#Look at misclassified cases
credit14.complete$predictedCreditability<-credit14.lda.predict$class
credit14.complete[credit14.complete$Creditability!=credit14.complete$predictedCreditability,]

# K-Nearest Neighbour (Method 4)


## store it as data frame
dia <- data.frame(credit14)

## create a random number equal 90% of total number of rows
set.seed(321)  # To get the same random sample
ran <- sample(1:nrow(dia),0.9 * nrow(dia)) #0.9 is parameter
##the normalization function is created
nor <-function(x) { (x -min(x))/(max(x)-min(x)) } # function
## run nomalization on the chosen 6 columns of dataset because they are the predictors
dia_norm <- as.data.frame(lapply(dia[,c(3,6,12,14,17,19)], nor))
summary(dia_norm) #summary, fyi only

## extract training dataset
dia_train <- dia_norm[ran,]
## extract testing dataset
dia_test <- dia_norm[-ran,]
## the 1st column of training dataset because that is what we need to predict about testing dataset
dia_target <- as.factor(dia[ran,1])
## the actual values of 1st column of testing dataset to compaire it with values that will be predicted
test_target <- as.factor(dia[-ran,1])

## run K-NN function
pr <- knn(dia_train,dia_test,cl=dia_target,k=21) #k is parameter
pr
## create the confucion matrix
tab <- table(pr,test_target)
tab
## check the accuracy
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100} # function
accuracy(tab)
























