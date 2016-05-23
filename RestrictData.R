patelSubtype = as.character(annots$patel$SubType)
patelKeepers = rownames(annots$patel)[
        patelSubtype %in% c("subtype: Mes", "subtype: Pro")]

xs$patel = xs$patel[patelKeepers, ]
xnorms$patel = xnorms$patel[patelKeepers, ]
annots$patel = droplevels(annots$patel[patelKeepers, ])


montastierTime = as.character(annots$montastier$Time)
montastierKeepers = rownames(annots$montastier)[
        montastierTime %in% c("C1", "C2")]

xs$montastier = xs$montastier[montastierKeepers, ]
xnorms$montastier = xnorms$montastier[montastierKeepers, ]
annots$montastier = droplevels(annots$montastier[montastierKeepers, ])
