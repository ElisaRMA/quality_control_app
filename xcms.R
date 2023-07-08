
suppressPackageStartupMessages(library(xcms))
suppressPackageStartupMessages(library(CAMERA))


xset <- xcmsSet( 
        method   = "matchedFilter",
        fwhm     = 10.78,
        snthresh = 1,
        step     = 1,
        steps    = 9,
        sigma    = 4.57788347205708,
        max      = 10,
        mzdiff   = 1,#-8.2
        index    = FALSE)

print('part 1 - complete')

xset2 <- retcor( 
        xset,
        method         = "obiwarp",
        plottype       = "none",
        distFunc       = "cor_opt",
        profStep       = 1,
        #center         = NULL, #537
        response       = 1,
        gapInit        = 0.2,
        gapExtend      = 2.4,
        factorDiag     = 2,
        factorGap      = 1,
        localAlignment = 0)


print('part 2 - complete')

xset3 <- group( 
        xset2,
        method  = "density",
        bw      = 29.2,
        mzwid   = 0.3,#0.035
        minfrac = 0.1, #0.7
        minsamp = 50, #1
        max     = 100)


print('part 3 - complete')

xset4 <- fillPeaks(xset3)

an <- xsAnnotate(xset4)
#Creation of an xsAnnotate object

anF <- groupFWHM(an, perfwhm = 0.6)

#Perfwhm = parameter defines the window width, which is used for matching
anI <- findIsotopes(anF, mzabs=0.01)

#Mzabs = the allowed m/z error
anIC <- groupCorr(anI, cor_eic_th=0.1)

anFA <- findAdducts(anIC, polarity="negative") #change polarity accordingly

print('creating dataset...')

write.csv(getPeaklist(anIC), file="data.csv") # generates a table of features
