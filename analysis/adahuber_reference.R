# Native adaHuber and an explicitly adapted perspective block solver.
# Numerical source in the installed package is unchanged.
Sys.setenv(OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1)
.libPaths(c(normalizePath('analysis/outputs/r-library'), .libPaths()))
suppressPackageStartupMessages(library(adaHuber))
args <- commandArgs(trailingOnly=TRUE)
directory <- args[1]
X <- as.matrix(read.table(file.path(directory,'X.tsv'),sep='\t'))
y <- scan(file.path(directory,'y.tsv'),quiet=TRUE)
controls <- read.delim(file.path(directory,'controls.tsv'),header=TRUE)
cfg <- setNames(controls$value,controls$key)
n <- length(y); p <- ncol(X); Z <- cbind(1,X)
rho <- cfg[['rho']]; a <- cfg[['a']]; lambda0 <- cfg[['lambda0']]
sigma0 <- cfg[['sigma0']]; tolerance <- cfg[['tolerance']]
inner_tolerance <- cfg[['inner_tolerance']]; max_outer <- as.integer(cfg[['max_outer']])
coefficient_batch <- as.integer(cfg[['coefficient_batch']])
max_inner <- as.integer(cfg[['max_inner']]); root_tolerance <- cfg[['root_tolerance']]
phi0 <- cfg[['phi0']]; gamma <- cfg[['gamma']]; repeats <- as.integer(cfg[['repeats']])
batch <- as.integer(cfg[['timing_batch']])
root_max <- as.integer(cfg[['root_max']]); check_every <- as.integer(cfg[['inner_check_every']])
soft <- function(z,t) sign(z)*pmax(abs(z)-t,0)
score <- function(residual,sigma) sign(residual)*pmin(abs(residual)/sigma,rho)
quality <- function(beta,sigma) {
  psi <- score(as.vector(y-Z%*%beta),sigma)
  g <- -as.vector(crossprod(Z,psi))/n
  pen <- c(0,rep(lambda0,p))
  c(coefficient=max(abs(beta-soft(beta-g,pen))),scale=abs(a-mean(psi^2)/2))
}
scale_root <- function(residual) {
  if (all(residual==0) || a>=rho^2*sum(residual!=0)/(2*n)) return(0)
  upper <- sqrt(mean(residual^2)/(2*a))
  if(max(abs(residual))<=rho*upper) return(upper)
  f <- function(s) if(s==0) a-rho^2*sum(residual!=0)/(2*n) else a-mean(pmin((residual/s)^2,rho^2))/2
  while (f(upper)<0) upper <- upper*cfg[['bracket_expansion']]
  uniroot(f,c(0,upper),tol=root_tolerance,maxiter=root_max)$root
}
# lamm is the compiled proximal-gradient coefficient step shipped in adaHuber.
lamm <- getFromNamespace('lamm','adaHuber')
native_core <- getFromNamespace('adaHuberLasso','adaHuber')
eta0 <- sigma0*lambda0
native_public <- function() adaHuber.lasso(X,y,lambda=eta0,tau=0,phi0=phi0,
                                          gamma=gamma,epsilon=inner_tolerance,iteMax=max_inner)
native <- function() {
  beta <- native_core(Z,y,eta0,p,phi0,gamma,inner_tolerance,max_inner)
  # In release 1.1, after its first nontrivial iteration tau is a function of Y.
  tau <- sqrt(getFromNamespace('rootf1','adaHuber')(y^2,n,log(n*p)/n,min(y^2),sum(y^2)))
  list(beta=as.vector(beta),sigma=tau/rho,iterations=NA_integer_)
}
adapted <- function(record=FALSE) {
  beta <- numeric(p+1);sigma <- sigma0;phi <- phi0;inner <- 0L
  trace_beta <- list(beta);trace_scale <- sigma
  for (outer in seq_len(max_outer)) {
    sigma <- scale_root(as.vector(y-Z%*%beta))
    if (sigma<=0) stop('Matched adapter requires positive scale for this benchmark')
    tau <- rho*sigma; penalty <- c(0,rep(sigma*lambda0,p))
    for (k in seq_len(coefficient_batch)) {
      # The package modifies its numeric beta argument in place. Force a fresh
      # vector each call to keep R aliases and trace snapshots independent.
      beta <- beta + 0
      phi <- max(phi0,lamm(Z,y,penalty,beta,tau,phi,gamma,p,1/n)/gamma)
      inner <- inner+1L
      if (k%%check_every==0L || k==coefficient_batch) {
        psi <- sign(y-Z%*%beta)*pmin(abs(y-Z%*%beta),tau)
        g <- -as.vector(crossprod(Z,psi))/n
        if (max(abs(beta-soft(beta-g,penalty)))/sigma<=inner_tolerance) break
      }
    }
    if (record) {trace_beta[[length(trace_beta)+1L]] <- beta+0;trace_scale <- c(trace_scale,sigma)}
    if (max(quality(beta,sigma))<=tolerance) break
  }
  list(beta=as.vector(beta),sigma=sigma,iterations=outer,inner_iterations=inner,
       trace_beta=if(record) do.call(rbind,trace_beta) else NULL,trace_scale=if(record) trace_scale else NULL)
}
# Shared data centering and sample-SD standardization match the public wrapper.
if (max(abs(colMeans(X)))>cfg[['standardization_tolerance']] || max(abs(apply(X,2,sd)-1))>cfg[['standardization_tolerance']]) stop('Inputs must already match R standardization')
public <- native_public();prepared <- native()
public_error <- max(abs(as.vector(public$coef)-prepared$beta),abs(public$tau/rho-prepared$sigma))
if (public_error>cfg[['wrapper_match_tolerance']]) stop('Prepared native call differs from public adaHuber')
# Warm R byte code, compiled entry points, BLAS and the numerical routines.
invisible(adapted());invisible(native());invisible(native_public())
rows <- list();solutions <- list()
for (rep in seq_len(repeats)) {
  methods <- if(rep%%2L) c('native','adapted') else c('adapted','native')
  for (method in methods) {
    elapsed <- system.time(for (call in seq_len(batch)) result <- if(method=='native') native() else adapted())
    rows[[length(rows)+1L]] <- data.frame(method=method,repeat_id=rep,
      cpu_seconds=unname(sum(elapsed[c('user.self','sys.self')]))/batch,wall_seconds=unname(elapsed['elapsed'])/batch,
      sigma=result$sigma,iterations=result$iterations,
      inner_iterations=if(is.null(result$inner_iterations)) NA_integer_ else result$inner_iterations)
    solutions[[paste(method,rep,sep='-')]] <- result$beta
  }
  elapsed <- system.time(for (call in seq_len(batch)) invisible(native_public()))
  rows[[length(rows)+1L]] <- data.frame(method='native_public',repeat_id=rep,
    cpu_seconds=unname(sum(elapsed[c('user.self','sys.self')]))/batch,wall_seconds=unname(elapsed['elapsed'])/batch,
    sigma=public$tau/rho,iterations=public$iteration,inner_iterations=NA_integer_)
}
write.table(do.call(rbind,rows),file.path(directory,'r-timings.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
write.table(do.call(cbind,solutions),file.path(directory,'r-coefficients.tsv'),sep='\t',row.names=FALSE,col.names=TRUE,quote=FALSE)
trace <- adapted(TRUE)
write.table(cbind(sigma=trace$trace_scale,trace$trace_beta),file.path(directory,'r-trace.tsv'),sep='\t',row.names=FALSE,col.names=TRUE,quote=FALSE)
writeLines(c(paste('adaHuber',as.character(packageVersion('adaHuber'))),R.version.string,
  paste('Rcpp',as.character(packageVersion('Rcpp'))),paste('RcppArmadillo',as.character(packageVersion('RcppArmadillo'))),
  paste('public_prepared_max_difference',format(public_error,digits=17)),capture.output(sessionInfo())),file.path(directory,'r-session.txt'))
