.libPaths(c(normalizePath('analysis/outputs/r-library'), .libPaths()))
suppressPackageStartupMessages(library(rome))
x <- matrix(1, 3, 1)
y <- c(-1, .2, .5)
rows <- lapply(c('none', 'adaptive'), function(screen) {
  fit <- rome.adaptive(x=x, y=y, lambda=.1, delta=.5, alpha=1,
                      adapt=FALSE, intercept=FALSE, preprocess='none',
                      centering=FALSE, screen=screen, KKT=(screen!='none'),
                      eps=1e-12, max.iter=100)
  data.frame(screen=screen, coefficient=as.numeric(fit$beta[1,1]))
})
write.csv(do.call(rbind,rows), commandArgs(trailingOnly=TRUE)[1], row.names=FALSE)
