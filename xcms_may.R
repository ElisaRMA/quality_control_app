options(warn=-1)
suppressMessages(library(xcms))
suppressMessages(library(CAMERA))

xset <- xcmsSet( 
        method   = "matchedFilter",
        fwhm     = 29.4,
        snthresh = 16.1595968, #16.1595968
        step     = 1,
        steps    = 12,
        sigma    = (29.4/2.3548), #12.4851367419738,
        max      = 5,
        mzdiff   = -11, # -11 WAS THE STANDARD
        index    = FALSE)

xset2 <- retcor( 
        xset,
        method         = "obiwarp",
        plottype       = "none",
        distFunc       = "cor_opt",
        profStep       = 1,
        response       = 1,
        gapInit        = 0.26,
        gapExtend      = 2.1,
        factorDiag     = 2,
        factorGap      = 1,
        localAlignment = 0)


xset3 <- group( 
        xset2,
        method  = "density",
        bw      = 50,
        mzwid   = 1,
        minfrac = 0.1,
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
anIC <- groupCorr(anI, cor_eic_th=0.1)

anFA <- findAdducts(anIC, polarity="negative") #change polarity accordingly

data_processed = getPeaklist(anIC)

write.csv(getPeaklist(anIC), file="testing_app.csv") # generates a table of features




