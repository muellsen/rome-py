# Base R only; source selection follows rome Section 4.5 (1-based columns).
load('sources/data/glass.RData')
write.table(x[,15:500], 'sources/data/glass-X.tsv', row.names=FALSE,
            col.names=FALSE, sep='\t')
write.table(y$PbO, 'sources/data/glass-y.tsv', row.names=FALSE,
            col.names=FALSE, sep='\t')
