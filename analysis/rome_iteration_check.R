args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(normalizePath('analysis/outputs/r-library'), .libPaths()))
suppressPackageStartupMessages(library(rome))
directory <- args[1]
x <- as.matrix(read.table(file.path(directory,'X.tsv')))
y <- scan(file.path(directory,'y.tsv'),quiet=TRUE)
lambdas <- scan(file.path(directory,'lambda.tsv'),quiet=TRUE)
fit <- rome.adaptive(x=x,y=y,lambda=lambdas,delta=.5,alpha=1,adapt=FALSE,
                    intercept=FALSE,preprocess='none',centering=FALSE,
                    screen='adaptive',KKT=TRUE,eps=1e-4,max.iter=100000)
previous <- as.matrix(read.table(file.path(directory,'rome-beta.tsv')))
difference <- max(abs(t(fit$beta)-previous))
if (difference > 1e-12*max(1,max(abs(previous)))) stop('Diagnostic R trajectory differs')
write.csv(data.frame(lambda=fit$lambda,iterations=fit$iter,
                     nonzeros=colSums(fit$beta!=0),maximum_beta_difference=difference),
          args[2],row.names=FALSE)
