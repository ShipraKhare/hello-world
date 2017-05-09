# Load package arules
library(arules)
library(arulesViz)

# List datasets in package
#data()
#load dataset 
transactions <- read.transactions("C:/Users/team/Desktop/DataMining/Assignment/Assignment1/transactions.txt",format="single",sep=",",cols=c(1,2))
# Type of data structure the data is ? 
class(transactions)
# summary showing basic statistics of the data set
summary(transactions)

# plot frequencies of frequent items in the dataset
itemFrequencyPlot(transactions, support=0.1, cex.names=0.8)

# Mine association rules using Apriori algorithm implemented in arules.
rules <- apriori(transactions, parameter = list(support= 0.01 , confidence= 0.2))
#summary of rules
summary(rules)
# Inspect rules
inspect(rules)
#inspect top 5 rules by highest lift
inspect(head(sort(rules, by ="lift"),5))

# Visualization of rules
#Plotting rules
plot(rules)
# Interactive plots for rules
sel <- plot(rules, measure=c("support", "lift"), shading="confidence", interactive=TRUE)

# Two key plot
plot(rules , shading="order", control=list(main="two-key plot"))
#Plotting RulesBev1
plot(RulesBev1, method="matrix", measure="lift", control=list(reorder=TRUE))


# 1.Purchase pattern related to beverages (Wine , Beer )
#Find subset of rules that has Wine on the right hand side

RulesBev1 <- subset(rules, subset = rhs %ain% "Wine")
summary(RulesBev1)
inspect(RulesBev1)


#Find subset of rules that has Wine and Beer in the left hand side.
RulesBev2 <- subset(rules, subset = lhs %ain%  "Wine"|lhs %ain%  "Beer" )
summary(RulesBev2)
inspect(RulesBev2)

#generating rules for beer on RHS from transactional data using apriori algorithm
beerRule<-apriori(data=transactions, parameter=list(supp=0.01,conf = 0.15,minlen=2), 
                  appearance = list(default="lhs",rhs="Beer"),
                  control = list(verbose=F))
#Sorting Beerrule by confidence in descending order
rules1<-sort(beerRule, decreasing=TRUE,by="confidence")
summary(rules1)
inspect(rules1)

# Visualization for 1st question subrules

# plot for subrules
plot(RulesBev1,method="graph",interactive=TRUE,shading=NA)
plot(RulesBev2,method="graph",interactive=TRUE,shading=NA)
plot(beerRule,method="graph",interactive=TRUE,shading=NA)

################################################################################
# 2.Pattern with respect to canned Vs fresh

#Subrules for Fresh Vegetables on the rhs
FreshRules <- subset(rules, subset = rhs %pin% "Fresh Vegetables")
summary(FreshRules)
inspect(FreshRules[1:20])

# Subrules for Fresh Fruit on the rhs
FreshRules1 <- subset(rules, subset = rhs %pin% "Fresh Fruit")
summary(FreshRules1)
inspect(FreshRules1[1:20])

#subrule for both Fresh Fruit and Fresh Vegetable on the lhs
FreshRules2 <- subset(rules, subset = lhs %ain% c("Fresh Fruit", "Fresh Vegetables"))
summary(FreshRules2)
inspect(FreshRules2)

#Subrule for fresh Vegetable and Canned Vegetables on lhs.
cannedRules <- subset(rules, subset = lhs %ain% c("Fresh Vegetables", "Canned Vegetables"))
summary(cannedRules)
inspect(cannedRules)

#visualization for 2nd question
#plotting first 20 subrules with high lift for fresh vegetables on rhs
subrules2 <- head(sort(FreshRules, by="lift"), 20)
plot(subrules2, method="graph")

#plotting subrule for fresh fruit on rhs 
plot(FreshRules1,method="graph",interactive=TRUE,shading=NA)

#plotting subrule for fresh fruit and fresh vegetables on lhs
plot(FreshRules2,method="graph",interactive=TRUE,shading=NA)

#Plot for comparision of fresh vegetables and canned vegetables
subrules3 <- head(sort(cannedRules, by="lift"), 10)
plot(subrules3,method="graph",interactive=TRUE,shading=NA)

########################################################
# 3. Small and Large transaction
#Subrule for small baskets with item less than or equal to 2
rulesSmallSize <- subset(rules, subset = size(rules) <=2 )
#summary for ruleSmallSize
summary(rulesSmallSize)
inspect(rulesSmallSize)

#Subrule for Large baskets with item more than or equal to 5
rulesLargeSize <- subset(rules, subset = size(rules) >= 5 )
summary(rulesLargeSize)
inspect(rulesLargeSize)
inspect(head(sort(rulesLargeSize, by ="lift"),5))

# Visualization for question 3 
#plotting rulesSmallSize for small item basket
plot(rulesSmallSize, method="paracoord")

# Interactive plot for rulesSmallSize
#sel <- plot(rulesSmallSize, measure=c("support", "lift"), shading="confidence", interactive=TRUE)
# plotting large itemset
plot(rulesLargeSize, method="paracoord")
#Interactice plot rulesLargeSize
#sel <- plot(rulesLargeSize, measure=c("support", "lift"), shading="confidence", interactive=TRUE)

#########################################################
# 4.One more intresting pattern:Milk and Cereal
#  Subsets. find subset of rules that has Milk on the Rhs and Cereal on lhs
Rulesinterest1 <- subset(rules, subset = rhs %pin%  "Milk" & lhs %ain% "Cereal")
#Summary of Rulesinterest1
summary(Rulesinterest1)
inspect(Rulesinterest1)

#Subsets. find subset of rules that has Milk on the lhs and Cereal on rhs
Rulesinterest2 <- subset(rules, subset = lhs %ain%  "Milk" & rhs %ain% "Cereal")
summary(Rulesinterest2)
inspect(Rulesinterest2)

#Visualization for question 4;
#Plot for Rulesinterest1 that has Milk on the rhs and cereal on the lhs
plot(Rulesinterest1, method="paracoord")

#Plot for Rulesinterest2 that has milk on the lhs and cereal on the rhs
plot(Rulesinterest2, method="graph")






