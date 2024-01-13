library(xcms)
library(CAMERA)

xset <- xcmsSet( 
        method   = "matchedFilter",
        fwhm     = 28,#7.5
        snthresh = 3,
        step     = 1, # numeric(1) specifying the width of the bins/slices in m/z dimension
        steps    = 6, # numeric(1) defining the number of bins to be merged before filtration (i.e. the number of neighboring bins that will be joined to the slice in which filtration and peak detection will be performed).
        sigma    = 3.18498386274843,
        max      = 3,# 5
        mzdiff   = 1, 
        index    = FALSE)

xset2 <- retcor( 
        xset,
        method         = "obiwarp",
        plottype       = "none",
        distFunc       = "cor_opt",
        profStep       = 1,
        response       = 1,
        gapInit        = 0,
        gapExtend      = 2.7,
        factorDiag     = 2,
        factorGap      = 1,
        localAlignment = 0)

xset3 <- group( 
        xset2,
        method  = "density",
        bw      = 22,
        mzwid   = 1,
        minfrac = 0.3,
        minsamp = 1,
        max     = 50)

xset4 <- fillPeaks(xset3)
# The IPO script ends here

# Substitute the object names inside the ( ) accordingly.

an <- xsAnnotate(xset4)

#Creation of an xsAnnotate object
anF <- groupFWHM(an, perfwhm = 0.6)

#Perfwhm = parameter defines the window width, which is used for matching
anI <- findIsotopes(anF, mzabs=0.01)

#Mzabs = the allowed m/z error
anIC <- groupCorr(anI, cor_eic_th=0.75)
anFA <- findAdducts(anIC, polarity="negative") #change polarity accordingly

write.csv(getPeaklist(anIC), file='test.csv') # generates a table of features
