simulate2Group = function(n=100, p=1000,
        n1=ceiling(0.5*n), effect=rep(1, 10)) {
    x = matrix(rnorm(n*p), nrow=n, ncol=p)
    y = factor(c(rep("A", n1), rep("B", (n-n1))))
    colnames(x) = paste0("g", 1:p)
    rownames(x) = paste0("i", 1:n)
    names(y) = rownames(x)
    for (i in 1:length(effect)) {
        x[y=="B", i] = x[y=="B", i] + effect[i]
    }
    return(list(x=x, y=y))
}
