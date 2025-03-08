options(warn=-1)
if (!requireNamespace("BiocManager", quietly=TRUE))
        install.packages("BiocManager")

BiocManager::install(version = "1.30.16")
BiocManager::install("xcms", version = "1.30.16")
BiocManager::install("CAMERA", version = "1.30.16")

suppressMessages(library(xcms))
suppressMessages(library(CAMERA))

xset <- xcmsSet( 
        method   = "matchedFilter",
        fwhm     = 18, #29.4
        snthresh = 3, #16.1595968
        step     = 1,
        steps    = 12,#12
        sigma    = 12.48, #12.48
        max      = 3,#5
        mzdiff   = 1,#-8.2
        index    = FALSE)

xset2 <- retcor( 
        xset,
        method         = "obiwarp",
        plottype       = "none",
        distFunc       = "cor_opt",
        profStep       = 1,
        #center         = datafiles[625], #625, #625 train, 182 val
        response       = 1,
        gapInit        = 0.2,
        gapExtend      = 2.4,
        factorDiag     = 2,
        factorGap      = 1,
        localAlignment = 0)

xset3 <- group( 
        xset2,
        method  = "density",
        bw      = 29.2,
        mzwid   = 1,#0.035
        minfrac = 0.05, #0.7
        minsamp = 1, #50 (original. Changed to 1 to garantee will have something. Further filtering done later on)
        max     = 100)

xset4 <- fillPeaks(xset3)

# The IPO script ends here

# Substitute the object names inside the ( ) accordingly.

an <- xsAnnotate(xset4)
#Creation of an xsAnnotate object

anF <- groupFWHM(an, perfwhm = 0.6)

#Perfwhm = parameter defines the window width, which is used for matching
anI <- findIsotopes(anF, mzabs=0.01)

#Mzabs = the allowed m/z error
anIC <- groupCorr(anI, cor_eic_th=0.1)

anFA <- findAdducts(anIC, polarity="negative") #change polarity accordingly

data_processed = getPeaklist(anIC)

write.csv(getPeaklist(anIC), file="testing_app.csv") # generates a table of features




