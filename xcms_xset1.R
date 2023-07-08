suppressPackageStartupMessages(library(xcms))
suppressPackageStartupMessages(library(CAMERA))
options(warn=-1)

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

