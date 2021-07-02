#TODO: Input is complex and contains numbers, should be generalized
args = commandArgs(trailingOnly=TRUE)
if (length(args)!=3) {
  stop("Usage: Rscript --vanilla Rprogram InDataDir dev_id libID", call.=FALSE)
}
InDataDir = args[1]
dev_id = as.integer(args[2])
libID= as.integer(args[3]) 

if (libID == 0){ # CuCuBLAS
	infile_h2d = sprintf("%s/cublasSet_Get_to-%d_from--1.log", InDataDir, dev_id)
	infile_d2h = sprintf("%s/cublasSet_Get_to--1_from-%d.log", InDataDir, dev_id)
} else {
	stop("Invalid libID", call.=FALSE)
}

h2d_dataset <- read.table(infile_h2d,  header = FALSE, sep = ",")
#summary(h2d_dataset)

simple_h2d_train_set <- subset(h2d_dataset, select=c("V1", "V3", "V6"))
simple_h2d_train_set$bytes <- c(I(simple_h2d_train_set$V1*simple_h2d_train_set$V1*8))
#summary(simple_h2d_train_set)

#Simple linear regression
regression_h2d <- lm(V3 ~ bytes , data = subset(simple_h2d_train_set))
#summary(regression_h2d)
inter_h2d <- regression_h2d$coefficients[1]
G_h2d <- regression_h2d$coefficients[2]

#Bidirectional linear regression - Calculate slowdown 
regression_h2d_bid <- lm(V6 ~ bytes , data = subset(simple_h2d_train_set))
slowdown_h2d <- regression_h2d_bid$coefficients[2]/ G_h2d

d2h_dataset <- read.table(infile_d2h,  header = FALSE, sep = ",")
summary(d2h_dataset)

simple_d2h_train_set <- subset(d2h_dataset, select=c("V1", "V3", "V6"))
simple_d2h_train_set$bytes <- c(I(simple_d2h_train_set$V1*simple_d2h_train_set$V1*8))
#summary(simple_d2h_train_set)

#Simple linear regression
regression_d2h <- lm(V3 ~ bytes , data = subset(simple_d2h_train_set))
#summary(regression_h2d)
inter_d2h <- regression_d2h$coefficients[1]
G_d2h <- regression_d2h$coefficients[2]

#Bidirectional linear regression - Calculate slowdown 
regression_d2h_bid <- lm(V6 ~ bytes , data = subset(simple_d2h_train_set))
slowdown_d2h <- regression_d2h_bid$coefficients[2]/ G_d2h

cat("LogP H2D model:","\n")
cat("Intercept = ", inter_h2d,"\n")
cat("Coefficient =", G_h2d,"\n")
cat("Slowdown =", slowdown_h2d,"\n")

cat("\nLogP D2H model:","\n")
cat("Intercept = ", inter_d2h,"\n")
cat("Coefficient =", G_d2h,"\n")
cat("Slowdown =", slowdown_d2h,"\n")

# Add calculated overhead as intercept
#h2d <- c("inter" = inter_h2d, "coef" = G_h2d$coefficients[1], "sl" = slowdown_h2d)
#d2h <- c("inter" = inter_d2h, "coef" = G_d2h$coefficients[1], "sl" = slowdown_d2h)
# Store for C use. 
#write.csv(data.frame(h2d, d2h) ,"../Models/transfer_model_dungani.log")

#outfile_h2d<-file("../Models/transfer_model_dungani_2_-1.log")
#outfile_d2h<-file("../Models/transfer_model_dungani_-1_2.log")
outfile_h2d_name = sprintf("%s/../Database/Linear-Model_to-%d_from--1.log", InDataDir, dev_id)
outfile_d2h_name = sprintf("%s/../Database/Linear-Model_to--1_from-%d.log", InDataDir, dev_id)
outfile_h2d<-file(outfile_h2d_name)
outfile_d2h<-file(outfile_d2h_name)

writeLines(c(toString(inter_h2d),toString(G_h2d), toString(slowdown_h2d)), outfile_h2d)
close(outfile_h2d)

writeLines(c(toString(inter_d2h),toString(G_d2h), toString(slowdown_d2h)), outfile_d2h)
close(outfile_d2h)
