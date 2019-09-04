Introduction to R-INLA for Spatial Analysis
============
The following presentation is to accompany this tutorial:
https://docs.google.com/presentation/d/1KRrZjiBx_UKETtN-EyEQxMWXAOUxlaMdTzqpPSPMxPc/edit#slide=id.g33fbbb4edf_0_14

Overview 
------
In this workshop we will be taking through an example of how to use the SPDE model in INLA package. We will analyse parasite prevalence data from Madagascar. The topics we will cover include:
* How to create a mesh for the continuously indexed spatial effects
* Implementing the SPDE model in R-INLA
* Conducting simple model selection and best fit model
* Spatial prediction using INLA and outside of INLA
* Model validation

For a much more thorough description of R-INLA and the details underlying the SPDE-models see: www.math.ntnu.no/inla/r-inla.org/papers/jss/lindgren.pdf For more details on the example we study here, see: www.math.ntnu.no/inla/r-inla.org/tutorials/spde/spde-tutorial.pdf

### Data used
Malaria prevalence data: Open-access malaria data hosted by the Malaria Atlas Project https://map.ox.ac.uk/ accessed using the malariaAtlas R package

Covariate data: a suite of satelitte imagery was cleaned and processed for this tutorial but is available upon request from the MAP team.

For data cleaning and prep work please run the R-Script **data_prep.R** 
```{r}
source('data_prep.R')
```
we reccomend looking at the script but it is not necessary to run as we provide the cleaned output for this workshop in ```inputs/MDG_clean.Rdata```


for this workshop we have included in the scripts the library packages needed. These are:

```{r}
library(INLA)
library(malariaAtlas)
library(raster)
library(sp)
library(tidyverse)
library(lattice)     
library(gridExtra)
```
for installation please use
```{r}
packages <- c("malariaAtlas", "raster", "sp", "tidyverse",
              "lattice", "gridExtra", "devtools", "rlang")
if(length(setdiff(packages, rownames(installed.packages()))) > 0) { 
  install.packages(setdiff(packages, rownames(installed.packages()))) }

#For INLA!!
if(length(setdiff("INLA", rownames(installed.packages()))) > 0){
  install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
}

```

Some useful resources to get more indepth with INLA are:
* The R-INLA webpage : http://www.r-inla.org/
* Spatial and Spatio-temporal Bayesian Models with R-INLA: https://sites.google.com/a/r-inla.org/stbook/
* Bayesian inference with INLA and R-INLA: https://becarioprecario.bitbucket.io/inla-gitbook/index.html
